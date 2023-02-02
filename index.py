#! /usr/bin/env python3


import csv
import logging
from pathlib import Path
from queue import Queue
from threading import Thread

import faiss
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def _make_index(q: Queue, emb_dim: int, dataset_config: DictConfig) -> None:
    index_out = Path.cwd() / "index.bin"
    doc_ids_out = Path.cwd() / "doc_ids.csv"

    index = faiss.IndexIDMap2(
        faiss.index_factory(emb_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    )
    doc_ids = {}
    dataset = instantiate(dataset_config, data_processor=None)

    while True:
        item = q.get()
        # sentinel
        if item is None:
            break

        out, ids = item
        index.add_with_ids(out, ids)
        doc_ids.update({id: dataset.get_orig_id(int(id)) for id in ids})

    LOGGER.info(f"writing {doc_ids_out}")
    with open(doc_ids_out, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("id", "orig_doc_id"))
        for id, orig_id in doc_ids.items():
            writer.writerow((id, orig_id))

    LOGGER.info(f"writing {index_out}")
    faiss.write_index(index, str(index_out))


@hydra.main(config_path="config", config_name="indexing", version_base="1.3")
def main(config: DictConfig) -> None:
    ranker = instantiate(config.ranker.model)
    ranker.load_state_dict(
        torch.load(config.ckpt_path, map_location=config.device)["state_dict"]
    )
    ranker = torch.nn.DataParallel(ranker)
    ranker.to(config.device)
    ranker.eval()
    dataset = instantiate(
        config.encoding_data, data_processor=instantiate(config.ranker.data_processor)
    )
    data_loader = instantiate(
        config.data_loader, dataset=dataset, collate_fn=dataset.collate_fn
    )

    q = Queue()
    t_index = Thread(
        target=_make_index,
        args=(q, ranker.module.embedding_dimension, config.encoding_data),
    )
    t_index.start()

    for ids, d_inputs in tqdm(data_loader):
        ids = ids.numpy()
        with torch.no_grad():
            out = (
                ranker(
                    {k: v.to(config.device) for k, v in d_inputs.items()},
                    action="encode_docs",
                )
                .detach()
                .cpu()
                .numpy()
            )
        q.put((out, ids))

    # sentinel
    q.put(None)


if __name__ == "__main__":
    main()
