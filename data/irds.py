from typing import Iterable

import ir_datasets
from ranking_utils.model.data import DataProcessor

from data import EncodingDataset, EncodingInstance


class IRDSCorpusEncodingDataset(EncodingDataset):
    """Encoding dataset for corpora from the ir_datasets library."""

    def __init__(
        self,
        data_processor: DataProcessor,
        dataset: str,
        content_attributes: Iterable[str],
        max_len: int = None,
    ) -> None:
        """Constructor.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
            dataset (str): Dataset identifier within ir_datasets (must have a corpus).
            content_attributes (Iterable[str]): Attributes of the document representation to use as content (dataset-specific).
            max_len (int, optional): Split long documents into chunks (length in characters). Defaults to None.
        """
        super().__init__(data_processor, max_len)
        self.dataset = ir_datasets.load(dataset)
        self.content_attributes = content_attributes

    def get_orig_id(self, int_id: int) -> str:
        return self.dataset.docs_iter()[int_id].doc_id

    def _get_data(self) -> Iterable[EncodingInstance]:
        for index, doc in enumerate(self.dataset.docs_iter()):
            content = [getattr(doc, attr) for attr in self.content_attributes]
            yield index, ". ".join(content)
