import abc
from typing import Any, Iterable, Tuple

import torch
from ranking_utils.model.data import DataProcessor
from torch.utils.data import IterableDataset

EncodingInstance = Tuple[int, str]
EncodingModelInput = Any
EncodingInput = Tuple[int, EncodingModelInput]
EncodingModelBatch = Any
EncodingBatch = Tuple[torch.Tensor, EncodingModelBatch]


class EncodingDataset(IterableDataset, abc.ABC):
    """PyTorch dataset for encoding using dual-encoder models."""

    def __init__(self, data_processor: DataProcessor, max_len: int = None) -> None:
        """Constructor.

        Args:
            data_processor (DataProcessor): A model-specific data processor.
            max_len (int, optional): Split long documents into chunks (length in characters). Defaults to None.
        """
        super().__init__()
        self.data_processor = data_processor
        self.max_len = max_len

    @abc.abstractmethod
    def get_orig_id(self, int_id: int) -> str:
        """Return the original ID of an item (i.e. query or document).

        Args:
            int_id (int): The internal integer ID.

        Returns:
            str: The original ID.
        """
        pass

    @abc.abstractmethod
    def _get_data(self) -> Iterable[EncodingInstance]:
        """An iterator over all encoding instances (e.g. the whole corpus).

        Yields:
            EncodingInstance: An item to be encoded.
        """
        pass

    def _split(self, item: str) -> Iterable[str]:
        """Split an item into chunks if "max_len" was set.

        Args:
            item (str): The input item.

        Yields:
            str: Consecutive chunks of the input.
        """
        if self.max_len is None:
            yield item
        else:
            for i in range(0, len(item), self.max_len):
                yield item[i : i + self.max_len]

    def __iter__(self) -> Iterable[EncodingInput]:
        """Yield the encoding inputs for the active worker only.

        Yields:
            EncodingInput: An input for the encoder.
        """
        worker_info = torch.utils.data.get_worker_info()

        for i, (index, item) in enumerate(self._get_data()):

            # if this belongs to another worker, skip
            if (
                worker_info is not None
                and i % worker_info.num_workers != worker_info.id
            ):
                continue
            for s in self._split(item):
                yield index, self.data_processor.get_encoding_input(s)

    def collate_fn(self, inputs: Iterable[EncodingInput]) -> EncodingBatch:
        """Collate inputs into a batch.

        Args:
            inputs (Iterable[EncodingInput]): The inputs.

        Returns:
            EncodingBatch: The resulting batch.
        """
        indices, inputs = zip(*inputs)
        return (
            torch.IntTensor(indices),
            self.data_processor.get_encoding_batch(inputs),
        )
