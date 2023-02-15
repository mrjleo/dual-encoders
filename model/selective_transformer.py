import logging
from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import AutoModel

from model import Encoder
from model.transformer import EncodingModelBatch

LOGGER = logging.getLogger(__name__)


class SelectiveTransformerEncoder(Encoder):
    """Encodes a string using the CLS token of Transformer models.
    Only a percentage of the input tokens are kept, controlled by "delta".
    The lowest-scoring tokens are dropped.
    """

    def __init__(
        self,
        pretrained_model: str,
        dropout: float = 0.1,
        delta: float = 0.7,
        weights: Union[Path, str] = None,
    ):
        """Constructor.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            delta (float, optional): Percentage of tokens to keep. Defaults to 0.7.
            weights (Union[Path, str], optional): Initial weights to load. Defaults to None.
        """
        super().__init__()
        self.delta = delta

        self.model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)

        dim = self.model.config.hidden_size
        self.selector = torch.nn.Sequential(
            torch.nn.Linear(dim, dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // 2, dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // 2, dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // 2, 1),
            torch.nn.Sigmoid(),
        )

        # the selector is always frozen during fine-tuning
        for p in self.selector.parameters():
            p.requires_grad = False

        if weights is not None:
            LOGGER.info("loading state dict from %s", weights)
            self.load_state_dict(torch.load(weights)["state_dict"])

    def _extract_fill_batch(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a batch and its scores as input. Reduces the batch length to (delta * orig_len).

        Args:
            inputs_embeds (torch.Tensor): Input token embeddings.
            attention_mask (torch.Tensor): Input attention mask.
            token_type_ids (torch.Tensor): Input token type IDs.
            scores (torch.Tensor): Corresponding scores.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Embeddings, attention mask, token type IDs, position IDs.
        """
        cur_batch_size = list(attention_mask.size())[0]
        cur_batch_length = list(attention_mask.size())[1]
        max_length = round(cur_batch_length * self.delta)

        position_ids = torch.stack(
            [
                torch.arange(
                    0, cur_batch_length, device=inputs_embeds.device, dtype=torch.long
                )
                for _ in range(cur_batch_size)
            ]
        )
        top_scores = torch.topk(scores.squeeze(dim=2), max_length, dim=1, sorted=False)
        index, _ = torch.sort(top_scores[1])

        mask = torch.zeros(
            cur_batch_size, cur_batch_length, device=inputs_embeds.device
        )
        mask = mask.scatter_(1, index, 1).bool()

        inputs_embeds = inputs_embeds[
            mask.unsqueeze(2).repeat(1, 1, self.model.config.hidden_size)
        ].reshape(cur_batch_size, -1, self.model.config.hidden_size)

        attention_mask = attention_mask[mask].reshape(cur_batch_size, -1)
        token_type_ids = token_type_ids[mask].reshape(cur_batch_size, -1)
        position_ids = position_ids[mask].reshape(cur_batch_size, -1)

        return inputs_embeds, attention_mask, token_type_ids, position_ids

    def forward(self, batch: EncodingModelBatch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        word_embeddings = self.model.get_input_embeddings()
        inputs_embeds = word_embeddings(input_ids)

        (
            inputs_embeds,
            attention_mask,
            token_type_ids,
            position_ids,
        ) = self._extract_fill_batch(
            inputs_embeds, attention_mask, token_type_ids, self.selector(inputs_embeds)
        )

        cls_out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )["last_hidden_state"][:, 0]

        return self.dropout(cls_out)

    @property
    def embedding_dimension(self) -> int:
        return self.model.config.hidden_size
