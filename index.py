#! /usr/bin/env python3


import logging
from queue import Queue
from threading import Thread

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from util import StandaloneEncoder

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="indexing", version_base="1.3")
def main(config: DictConfig) -> None:
    LOGGER.info("loading %s", config.ckpt_path)
    doc_encoder = StandaloneEncoder(
        config.doc_encoder, config.ckpt_path, "doc_encoder", config.device
    )

    # during indexing, the query tokenizer never gets called, so we can use None here
    dataset = instantiate(
        config.encoding_data,
        data_processor=instantiate(
            config.data_processor,
            query_tokenizer=None,
            doc_tokenizer=doc_encoder.tokenizer,
        ),
    )
    data_loader = instantiate(
        config.data_loader, dataset=dataset, collate_fn=dataset.collate_fn
    )
    index_writer = instantiate(config.index_writer)

    q = Queue()
    t_index = Thread(
        target=index_writer,
        args=(q,),
    )
    t_index.start()

    for ids, d_inputs in tqdm(data_loader):
        q.put((ids, doc_encoder._encode(d_inputs)))

    # sentinel
    q.put(None)

    t_index.join()
    index_writer.finalize_index()


if __name__ == "__main__":
    main()
