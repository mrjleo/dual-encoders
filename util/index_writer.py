import abc
import csv
import logging
from pathlib import Path
from queue import Queue
from typing import List

import faiss
from fast_forward import OnDiskIndex

LOGGER = logging.getLogger(__name__)


class IndexWriter(abc.ABC):
    """Abstract base class for index writers.

    Methods to be implemented:
        * __call__
        * save_index
    """

    def __init__(self, target_dir: Path = Path.cwd()) -> None:
        """Instantiate an index writer.

        Args:
            target_dir (Path, optional): The path where the index should be created. Defaults to Path.cwd().
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir = target_dir

    @abc.abstractmethod
    def __call__(self, q: Queue) -> None:
        """Construct the index using items from the queue.

        Args:
            q (Queue): Queue of document IDs and representations (Tuple[Sequence[str], torch.Tensor]).
        """
        pass

    @abc.abstractmethod
    def finalize_index(self) -> None:
        """Finalize the index on disk (called after indexing is complete)."""
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

    def finalize_index(self) -> None:
        index_out = self.target_dir / "index.bin"
        doc_ids_out = self.target_dir / "doc_ids.csv"

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

    index: OnDiskIndex = None

    def __call__(self, q: Queue) -> None:
        if self.index is None:
            ff_index_file = self.target_dir / "ff_index.h5"
            LOGGER.info("creating %s", ff_index_file)
            self.index = OnDiskIndex(ff_index_file)

        while True:
            item = q.get()

            # sentinel
            if item is None:
                break

            ids, out = item
            self.index.add(out, doc_ids=ids)

    def finalize_index(self) -> None:
        # nothing is required here
        pass
