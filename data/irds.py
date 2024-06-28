import logging
from pathlib import Path
from typing import Iterable, Union

import ir_datasets
from ranking_utils.datasets.trec import read_top_trec
from ranking_utils.model.data import DataProcessor

from data import EncodingDataset, EncodingInstance

LOGGER = logging.getLogger(__name__)


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
        self.dataset = dataset
        self.content_attributes = content_attributes

    def _get_data(self) -> Iterable[EncodingInstance]:
        for doc in ir_datasets.load(self.dataset).docs_iter():
            content = [getattr(doc, attr) for attr in self.content_attributes]
            yield doc.doc_id, ". ".join(content)


class IRDSPartialCorpusEncodingDataset(IRDSCorpusEncodingDataset):
    """Encoding dataset for corpora from the ir_datasets library.
    Encodes only documents that appear in a provided TREC runfile.
    """

    def __init__(
        self,
        data_processor: DataProcessor,
        dataset: str,
        trec_runfile: Union[Path, str],
        content_attributes: Iterable[str],
        max_len: int = None,
    ) -> None:
        """Constructor.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
            dataset (str): Dataset identifier within ir_datasets (must have a corpus).
            trec_runfile (Union[Path, str]): Only documents that appear in this TREC runfile will be yielded.
            content_attributes (Iterable[str]): Attributes of the document representation to use as content (dataset-specific).
            max_len (int, optional): Split long documents into chunks (length in characters). Defaults to None.
        """
        super().__init__(data_processor, dataset, content_attributes, max_len)

        # we support Path and str to make config with hydra easier
        self.doc_ids = set.union(*read_top_trec(Path(trec_runfile)).values())
        assert len(self.doc_ids) > 0
        LOGGER.info("encoding %s documents", len(self.doc_ids))

    def _get_data(self) -> Iterable[EncodingInstance]:
        for doc in (
            ir_datasets.load(self.dataset).docs_store().get_many_iter(self.doc_ids)
        ):
            content = [getattr(doc, attr) for attr in self.content_attributes]
            yield doc.doc_id, ". ".join(content)
