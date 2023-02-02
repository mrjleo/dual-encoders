from pathlib import Path
from typing import Iterable, Union

import h5py
from ranking_utils.model.data import DataProcessor

from data import EncodingDataset, EncodingInstance


class H5CorpusEncodingDataset(EncodingDataset):
    """Encoding dataset for pre-processed corpora (H5)."""

    def __init__(
        self,
        data_processor: DataProcessor,
        data_dir: Union[Path, str],
        max_len: int = None,
    ) -> None:
        """Constructor.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
            data_dir (Union[Path, str]): Directory that contains the pre-processed files (.h5).
            max_len (int, optional): Split long documents into chunks (length in characters). Defaults to None.
        """
        super().__init__(data_processor, max_len)
        self.data_file = Path(data_dir) / "data.h5"

    def get_orig_id(self, int_id: int) -> str:
        with h5py.File(self.data_file, "r") as fp:
            return fp["orig_doc_ids"].asstr()[int_id]

    def _get_data(self) -> Iterable[EncodingInstance]:
        with h5py.File(self.data_file, "r") as fp:
            yield from enumerate(fp["docs"].asstr())
