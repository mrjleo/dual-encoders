import csv
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import faiss
import numpy as np
import torch
from fast_forward.encoder import Encoder as FFEncoder
from hydra.utils import instantiate
from omegaconf import DictConfig


def batch_iter(it: Iterable, batch_size: int) -> Iterator:
    """Iterate in batches.

    Args:
        it (Iterable): Iterable to iterate over.
        batch_size (int): Batch size.

    Yields:
        Iterator: Batches of items.
    """
    it = iter(it)
    while True:
        l = list(islice(it, batch_size))
        if len(l) == 0:
            return
        yield l


def read_faiss_index(index_dir: Path) -> Tuple[faiss.Index, Dict[int, str]]:
    """Read a FAISS index created by the indexing script.

    Args:
        index_dir (Path): The directory where the index was saved.

    Returns:
        Tuple[faiss.Index, Dict[int, str]]: The index and integer IDs mapped to original string IDs.
    """
    index = faiss.read_index(str(index_dir / "index.bin"))

    orig_doc_ids = {}
    with open(index_dir / "doc_ids.csv", encoding="utf-8", newline="") as fp:
        for item in csv.DictReader(fp):
            orig_doc_ids[int(item["id"])] = item["orig_doc_id"]
    return index, orig_doc_ids


class StandaloneEncoder(FFEncoder):
    """Adapter class to use encoders for indexing, retrieval, or re-ranking."""

    def __init__(
        self,
        encoder_config: DictConfig,
        ckpt_file: Path,
        weights_prefix: str = "query_encoder",
        device: str = "cpu",
    ) -> None:
        """Constructor.

        Args:
            encoder_config (DictConfig): Encoder config.
            ckpt_file (Path): Checkpoint to load.
            weights_prefix (str, optional): Prefix of the keys to be loaded in the state dict of the checkpoint. Defaults to "query_encoder".
            device (str, optional): Device to use. Defaults to "cpu".
        """
        super().__init__()
        self.tokenizer = instantiate(encoder_config.tokenizer)
        self.encoder = instantiate(encoder_config.encoder)
        self.device = device

        self.encoder.to(device)
        sd_enc, sd_proj = {}, {}
        ckpt = torch.load(ckpt_file, map_location=device)
        for k, v in ckpt["state_dict"].items():

            # remove prefix and dot
            if k.startswith(weights_prefix):
                sd_enc[k[len(weights_prefix) + 1 :]] = v
            if k.startswith("projection"):
                sd_proj[k[11:]] = v

        self.encoder.load_state_dict(sd_enc)
        if ckpt["hyper_parameters"].get("projection_size") is not None:
            self.projection = torch.nn.Linear(
                self.encoder.embedding_dimension,
                ckpt["hyper_parameters"]["projection_size"],
            )
            self.projection.load_state_dict(sd_proj)
        else:
            self.projection = None
        self.encoder.eval()

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            rep = self.encoder(
                {k: v.to(self.device) for k, v in self.tokenizer(queries).items()}
            )
            if self.projection is not None:
                rep = self.projection(rep)
            return rep.detach().numpy()
