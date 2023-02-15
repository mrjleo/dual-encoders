import abc
import csv
import logging
from pathlib import Path
from queue import Queue
from typing import List

import faiss
import numpy as np
from fast_forward.index import InMemoryIndex

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
            q (Queue): Queue of tuples of document IDs and representations, i.e. Tuple[Sequence[str], torch.Tensor].
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

    index: faiss.Index = None
    doc_ids: List[str] = []

    def __call__(self, q: Queue) -> None:
        while True:
            item = q.get()

            # sentinel
            if item is None:
                break

            ids, out = item
            if self.index is None:
                self.index = faiss.index_factory(
                    out.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT
                )
            self.doc_ids.extend(ids)
            self.index.add(out)

    def save_index(self, target_dir: Path) -> None:
        index_out = target_dir / "index.bin"
        doc_ids_out = target_dir / "doc_ids.csv"

        LOGGER.info("writing %s", doc_ids_out)
        with open(doc_ids_out, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(("id", "orig_doc_id"))
            for id, orig_id in enumerate(self.doc_ids):
                writer.writerow((id, orig_id))

        LOGGER.info("writing %s", index_out)
        faiss.write_index(self.index, str(index_out))


class FastForwardIndexWriter(IndexWriter):
    """Writer for Fast-Forward indexes."""

    vectors: List[np.ndarray] = []
    doc_ids: List[str] = []

    def __call__(self, q: Queue) -> None:
        while True:
            item = q.get()

            # sentinel
            if item is None:
                break

            ids, out = item
            self.vectors.append(out)
            self.doc_ids.extend(ids)

    def save_index(self, target_dir: Path) -> None:
        index = InMemoryIndex()
        index.add(np.vstack(self.vectors), doc_ids=self.doc_ids)
        index_out = target_dir / "index.pkl"
        LOGGER.info("writing %s", index_out)
        index.save(index_out)
