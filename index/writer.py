import abc
import csv
import logging
from pathlib import Path
from queue import Queue

import faiss
import numpy as np
from fast_forward.index import InMemoryIndex

from data import EncodingDataset

LOGGER = logging.getLogger(__name__)


class IndexWriter(abc.ABC):
    """Abstract base class for index writers.
    Methods to be implemented:
        * __call__
        * save_index
    """

    @abc.abstractmethod
    def __call__(self, q: Queue) -> None:
        """Construct the index using items from the queue.

        Args:
            q (Queue): Queue of tuples of document representations and IDs, i.e. Tuple[Tensor, Sequence[str]].
        """
        pass

    @abc.abstractmethod
    def save_index(self, target_dir: Path) -> None:
        """Save the index on disk.

        Args:
            target_dir (Path): Target directory.
        """
        pass


class FAISSIndexWriter(IndexWriter):
    """Writer for FAISS indexes."""

    def __init__(self, emb_dim: int, dataset: EncodingDataset) -> None:
        """Constructor.

        Args:
            emb_dim (int): Dimension for vector representations.
            dataset (EncodingDataset): Encoding dataset to get document IDs from.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.dataset = dataset

        self.index = faiss.IndexIDMap2(
            faiss.index_factory(emb_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        )
        self.doc_ids = {}

    def __call__(self, q: Queue) -> None:
        while True:
            item = q.get()
            # sentinel
            if item is None:
                break

            out, ids = item
            self.index.add_with_ids(out, ids)
            self.doc_ids.update({id: self.dataset.get_orig_id(int(id)) for id in ids})

    def save_index(self, target_dir: Path) -> None:
        index_out = target_dir / "index.bin"
        doc_ids_out = target_dir / "doc_ids.csv"

        LOGGER.info(f"writing {doc_ids_out}")
        with open(doc_ids_out, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(("id", "orig_doc_id"))
            for id, orig_id in self.doc_ids.items():
                writer.writerow((id, orig_id))

        LOGGER.info(f"writing {index_out}")
        faiss.write_index(self.index, str(index_out))


class FastForwardIndexWriter(IndexWriter):
    """Factory for Fast-Forward indexes."""

    def __init__(self, emb_dim: int, dataset: EncodingDataset) -> None:
        """Constructor.

        Args:
            emb_dim (int): Dimension for vector representations.
            dataset (EncodingDataset): Encoding dataset to get document IDs from.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.dataset = dataset
        self.vectors = []
        self.doc_ids = []

    def __call__(self, q: Queue) -> None:
        while True:
            item = q.get()
            # sentinel
            if item is None:
                break

            out, ids = item
            self.vectors.append(out)
            self.doc_ids.extend([self.dataset.get_orig_id(int(id)) for id in ids])

    def save_index(self, target_dir: Path) -> None:
        index = InMemoryIndex()
        index.add(np.vstack(self.vectors), doc_ids=self.doc_ids)
        index_out = target_dir / "index.pkl"
        LOGGER.info(f"writing {index_out}")
        index.save(index_out)
