#! /usr/bin/env python3


import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import faiss
import hydra
import ir_datasets
import ir_measures
from fast_forward.encoder import QueryEncoder as FFQueryEncoder
from omegaconf import DictConfig
from ranking_utils import write_trec_eval_file
from tqdm import tqdm

from util import QueryEncoderAdapter, batch_iter, read_faiss_index

LOGGER = logging.getLogger(__name__)


def retrieve(
    encoder: FFQueryEncoder,
    index: faiss.Index,
    orig_doc_ids: Dict[int, str],
    dataset: ir_datasets.Dataset,
    k: int,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """Retrieve documents for the queries of a specified dataset.

    Args:
        encoder (FFQueryEncoder): The query encoder.
        index (faiss.Index): The index to retrieve from.
        orig_doc_ids (Dict[int, str]): Mapping of index IDs to original IDs.
        dataset (ir_datasets.Dataset): The dataset to take queries from.
        k (int): Number of documents to retrieve per query.
        batch_size (int): Query encoder batch size.

    Returns:
        Dict[str, Dict[str, float]]: The resulting TREC run.
    """
    all_queries = dataset.queries_iter()
    run = defaultdict(dict)
    for batch in tqdm(batch_iter(all_queries, batch_size)):
        ids, queries = zip(*batch)
        out = encoder.encode(queries)
        D, I = index.search(out, k)
        for q_id, doc_ids, scores in zip(ids, I, D):
            for doc_id, score in zip(doc_ids, scores):
                orig_doc_id = orig_doc_ids[doc_id]
                run[q_id][orig_doc_id] = float(score)
    return run


@hydra.main(config_path="config", config_name="retrieval", version_base="1.3")
def main(config: DictConfig) -> None:
    LOGGER.info(f"reading {config.ckpt_file}")
    encoder = QueryEncoderAdapter(config.query_encoder, config.ckpt_file, config.device)

    index_dir = Path(config.index_dir)
    LOGGER.info(f"reading {index_dir}")
    index, orig_doc_ids = read_faiss_index(index_dir)

    measures = list(map(ir_measures.parse_measure, config.metrics))
    with open(Path.cwd() / "metrics.csv", "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, ["type", "model_id", "dataset", "k"] + measures)
        writer.writeheader()

        dataset = ir_datasets.load(config.dataset)
        run = retrieve(
            encoder, index, orig_doc_ids, dataset, config.k, config.batch_size
        )
        write_trec_eval_file(Path(f"{config.model_id}.tsv"), run, config.model_id)

        row = ir_measures.calc_aggregate(measures, dataset.qrels_iter(), run)
        row["type"] = "retrieval"
        row["model_id"] = config.model_id
        row["dataset"] = config.dataset
        row["k"] = config.k
        writer.writerow(row)


if __name__ == "__main__":
    main()
