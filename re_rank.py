#! /usr/bin/env python3


import csv
import logging
from pathlib import Path

import hydra
import ir_datasets
import ir_measures
from fast_forward.ranking import Ranking
from hydra.utils import call
from omegaconf import DictConfig

from util import QueryEncoderAdapter

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="re_ranking", version_base="1.3")
def main(config: DictConfig) -> None:
    LOGGER.info(f"loading {config.ckpt_file}")
    adapter = QueryEncoderAdapter(config.query_encoder, config.ckpt_file, config.device)

    LOGGER.info("loading index")
    ff_index = call(config.ff_index_reader, query_encoder=adapter)

    LOGGER.info(f"reading {config.sparse_scores_file}")
    sparse_scores = Ranking.from_file(Path(config.sparse_scores_file))
    if config.cutoff_sparse is not None:
        sparse_scores.cut(config.cutoff_sparse)

    LOGGER.info(f"loading {config.dataset}")
    dataset = ir_datasets.load(config.dataset)
    queries = {query.query_id: query.text for query in dataset.queries_iter()}

    LOGGER.info("computing scores")
    ff_result = ff_index.get_scores(
        sparse_scores,
        queries,
        config.alpha,
        config.cutoff,
        config.early_stopping,
    )

    measures = list(map(ir_measures.parse_measure, config.metrics))
    with open(Path.cwd() / "metrics.csv", "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, ["type", "model_id", "dataset", "alpha", "cutoff"] + measures
        )
        writer.writeheader()
        for alpha, ranking in ff_result.items():
            ranking.name = f"{config.model_id}_{str(alpha)}"
            target = Path(f"{ranking.name}.tsv")
            LOGGER.info(f"writing {target}")
            ranking.save(target)

            row = ir_measures.calc_aggregate(
                measures, dataset.qrels_iter(), ranking.run
            )
            row["type"] = "fast-forward"
            row["model_id"] = config.model_id
            row["dataset"] = config.dataset
            row["alpha"] = alpha
            row["cutoff"] = config.cutoff
            writer.writerow(row)


if __name__ == "__main__":
    main()
