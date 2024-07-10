from typing import Dict, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from model import Encoder, Tokenizer

EncodingModelBatch = Dict[str, torch.LongTensor]


class TransformerTokenizer(Tokenizer):
    """Tokenizer for Transformer models."""

    def __init__(self, pretrained_model: str, max_length: int = None) -> None:
        """Constuctor.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub.
            max_length (int, optional): Maximum number of tokens. Defaults to None.
        """
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length

    def __call__(self, batch: Sequence[str]) -> EncodingModelBatch:
        """Tokenize a batch of strings.

        Args:
            batch (Sequence[str]): The tokenizer inputs.

        Returns:
            EncodingModelBatch: The tokenized inputs.
        """
        return self.tok(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )


class TransformerEncoder(Encoder):
    """Encodes a string using the CLS token of Transformer models."""

    def __init__(self, pretrained_model: str, dense_layer_dim: int = None) -> None:
        """Instantiate a Transformer encoder.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub.
            dense_layer_dim (int, optional): Add a dense layer to adjust the embedding dimension. Defaults to None.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        if dense_layer_dim is not None:
            self.dense = torch.nn.Linear(self.model.config.hidden_size, dense_layer_dim)
        else:
            self.dense = None

    def forward(self, batch: EncodingModelBatch) -> torch.Tensor:
        rep = self.model(**batch)["last_hidden_state"][:, 0]
        if self.dense is not None:
            return self.dense(rep)
        return rep

    @property
    def embedding_dimension(self) -> int:
        if self.dense is not None:
            return self.dense.out_features
        return self.model.config.hidden_size


class TransformerEmbeddingEncoder(Encoder):
    """Encodes a string using the average of the embedded tokens.
    Static token embeddings are obtained from a pre-trained Transformer model.
    """

    def __init__(self, pretrained_model: str, dense_layer_dim: int = None) -> None:
        """Constuctor.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub to get the token embeddings from.
            dense_layer_dim (int, optional): Add a dense layer to adjust the embedding dimension. Defaults to None.
        """
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        self.embeddings = model.get_input_embeddings()
        if dense_layer_dim is not None:
            self.dense = torch.nn.Linear(self.embeddings.embedding_dim, dense_layer_dim)
        else:
            self.dense = None

    def forward(self, batch: EncodingModelBatch) -> torch.Tensor:
        inputs = batch["input_ids"]
        lengths = (inputs != 0).sum(dim=1)
        sequences_emb = self.embeddings(inputs)

        # create a mask corresponding to sequence lengths
        _, max_len, emb_dim = sequences_emb.shape
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, emb_dim)

        # compute the mean for each sequence
        rep = torch.sum(mask * sequences_emb, dim=1) / lengths.unsqueeze(-1)

        if self.dense is not None:
            return self.dense(rep)
        return rep

    @property
    def embedding_dimension(self) -> int:
        if self.dense is not None:
            return self.dense.out_features
        return self.embeddings.embedding_dim
