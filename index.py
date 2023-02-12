#! /usr/bin/env python3


from pathlib import Path
from queue import Queue
from threading import Thread

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


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
    index_writer = instantiate(config.index_writer)

    q = Queue()
    t_index = Thread(
        target=index_writer,
        args=(q,),
    )
    t_index.start()

    for ids, d_inputs in tqdm(data_loader):
        d_inputs = {k: v.to(config.device) for k, v in d_inputs.items()}
        with torch.no_grad():
            out = ranker(d_inputs, action="encode_docs")
        q.put((ids, out.detach().cpu().numpy()))

    # sentinel
    q.put(None)

    t_index.join()
    index_writer.save_index(Path.cwd())


if __name__ == "__main__":
    main()
