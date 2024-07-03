#! /usr/bin/env python3


import csv
import logging
from collections import defaultdict
from pathlib import Path

import hydra
import ir_datasets
import ir_measures
from omegaconf import DictConfig
from ranking_utils import write_trec_eval_file
from tqdm import tqdm

from util import QueryEncoderAdapter, batch_iter, read_faiss_index

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="retrieval", version_base="1.3")
def main(config: DictConfig) -> None:
    LOGGER.info("loading %s", config.ckpt_path)
    encoder = QueryEncoderAdapter(config.query_encoder, config.ckpt_path, config.device)

    index_dir = Path(config.index_dir)
    LOGGER.info("loading %s", index_dir)
    index, orig_doc_ids = read_faiss_index(index_dir)

    LOGGER.info("loading %s", config.dataset)
    dataset = ir_datasets.load(config.dataset)

    run = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for batch in tqdm(batch_iter(dataset.queries_iter(), config.batch_size)):
        ids, queries = zip(*batch)
        out = encoder(queries)
        D, I = index.search(out, config.k)
        for q_id, doc_ids, scores in zip(ids, I, D):
            for doc_id, score in zip(doc_ids, scores):
                orig_doc_id = orig_doc_ids[doc_id]

                # maxP
                run[q_id][orig_doc_id] = max(float(score), run[q_id][orig_doc_id])

    run_file = Path.cwd() / f"retrieval_out_{config.name}.tsv"
    LOGGER.info("writing %s", run_file)
    write_trec_eval_file(run_file, run, config.name)

    if len(config.metrics) > 0:
        measures = list(map(ir_measures.parse_measure, config.metrics))
        LOGGER.info("computing %s", measures)
        results = ir_measures.calc_aggregate(measures, dataset.qrels_iter(), run)

        results_file = Path.cwd() / f"retrieval_results_{config.name}.csv"
        LOGGER.info("writing %s", results_file)
        with open(results_file, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, ["type", "name", "dataset", "k"] + measures)
            writer.writeheader()
            results["type"] = "retrieval"
            results["name"] = config.name
            results["dataset"] = config.dataset
            results["k"] = config.k
            writer.writerow(results)


if __name__ == "__main__":
    main()
