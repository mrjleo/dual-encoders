import abc
from typing import Any, Dict, Iterable, Sequence, Tuple, Union

import torch
from ranking_utils.model import Ranker, TrainingBatch, TrainingMode, ValTestBatch
from ranking_utils.model.data import DataProcessor
from transformers import get_constant_schedule_with_warmup

from model.metrics import KLDivergence, TensorStack

EncodingModelInput = str
EncodingModelBatch = Any

ModelInput = Tuple[EncodingModelInput, EncodingModelInput]
ModelBatch = Tuple[EncodingModelBatch, EncodingModelBatch]


class Tokenizer(abc.ABC):
    """Base class for tokenizers."""

    @abc.abstractmethod
    def __call__(self, batch: Sequence[str]) -> EncodingModelBatch:
        """Tokenize a batch of strings.

        Args:
            batch (Sequence[str]): The tokenizer inputs.

        Returns:
            EncodingModelBatch: The tokenized inputs.
        """
        pass


class Encoder(abc.ABC, torch.nn.Module):
    """Base class for encoders."""

    @abc.abstractmethod
    def forward(self, batch: EncodingModelBatch) -> torch.Tensor:
        """Encode a batch of inputs.

        Args:
            batch (EncodingBatch): The encoder model inputs.

        Returns:
            torch.Tensor: The encoded inputs.
        """
        pass

    @property
    @abc.abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The dimension of query or document vectors.
        """
        pass


class DualEncoderDataProcessor(DataProcessor):
    """Data processor for dual-encoder rankers."""

    def __init__(
        self,
        query_tokenizer: Tokenizer,
        doc_tokenizer: Tokenizer = None,
        char_limit: int = None,
    ) -> None:
        """Constructor. If "doc_tokenizer" is None, "query_tokenizer" will be used for documents.

        Args:
            query_tokenizer (Tokenizer): Tokenizer used for queries.
            doc_tokenizer (Tokenizer, optional): Tokenizer used for documents. Defaults to None.
            char_limit (int, optional): Maximum number of characters per query/document. Defaults to None.
        """
        super().__init__()
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer or query_tokenizer
        self.char_limit = char_limit

    def get_encoding_input(self, s: str) -> EncodingModelInput:
        """Sanitize an input string for encoding.

        Args:
            s (str): The input string.

        Returns:
            EncodingModelInput: The sanitized input string.
        """
        if len(s.strip()) == 0:
            s = "(empty)"
        return s[: self.char_limit]

    def get_model_input(self, query: str, doc: str) -> ModelInput:
        return self.get_encoding_input(query), self.get_encoding_input(doc)

    def get_encoding_batch(
        self, inputs: Iterable[EncodingModelInput], is_doc: bool = True
    ) -> EncodingModelBatch:
        """Return an input batch for an encoder.

        Args:
            inputs (Iterable[EncodingModelInput]): The inputs.
            is_doc (bool, optional): Whether to use the document or query tokenizer. Defaults to True.

        Returns:
            EncodingModelBatch: The resulting encoder batch.
        """
        if not isinstance(inputs, list):
            inputs = list(inputs)
        return self.doc_tokenizer(inputs) if is_doc else self.query_tokenizer(inputs)

    def get_model_batch(self, inputs: Iterable[ModelInput]) -> ModelBatch:
        queries, docs = zip(*inputs)
        return (
            self.get_encoding_batch(queries, False),
            self.get_encoding_batch(docs, True),
        )


class DualEncoder(Ranker):
    """Dual-encoder ranking/retrieval model."""

    def __init__(
        self,
        lr: float,
        warmup_steps: int,
        temperature: float,
        hparams: Dict[str, Any],
        query_encoder: Encoder,
        doc_encoder: Encoder = None,
        freeze_doc_encoder: bool = False,
        visualize_embeddings: bool = False,
        compute_kl_div: bool = False,
        num_embeddings: int = None,
    ):
        """Constructor. If "doc_encoder" is None, "query_encoder" will be used for documents.

        Args:
            lr (float): Learning rate.
            warmup_steps (int): Number of warmup steps.
            temperature (float): Contrastive loss temperature.
            hparams (Dict[str, Any]): Model hyperparameters.
            query_encoder (Encoder): Query encoder.
            doc_encoder (Encoder, optional): Document encoder. Defaults to None.
            freeze_doc_encoder (bool, optional): Freeze the weights of the document encoder. Defaults to False.
            visualize_embeddings (bool, optional): Visualize validation embeddings. Defaults to False.
            compute_kl_div (bool, optional): Compute KL divergence of validation embeddings. Defaults to False.
            num_embeddings (int, optional): Maximum number of validation embeddings computed. Defaults to None.
        """
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder or query_encoder
        assert (
            self.query_encoder.embedding_dimension
            == self.doc_encoder.embedding_dimension
        )

        self.projection = (
            torch.nn.Linear(
                in_features=self.doc_encoder.embedding_dimension,
                out_features=hparams["projection_size"],
            )
            if hparams.get("projection_size") is not None
            else None
        )
        self.dropout = torch.nn.Dropout(hparams["dropout"])
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.temperature = temperature
        self.save_hyperparameters(hparams)

        if freeze_doc_encoder:
            for p in self.doc_encoder.parameters():
                p.requires_grad = False

        self.visualize_embeddings = visualize_embeddings
        self.compute_kl_div = compute_kl_div
        self.kl_div = KLDivergence(self.embedding_dimension)
        self.q_enc_embeddings = TensorStack(self.embedding_dimension, num_embeddings)
        self.d_enc_embeddings = TensorStack(self.embedding_dimension, num_embeddings)

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The query and document vector dimension.
        """
        if self.projection is not None:
            return self.projection.out_features
        return self.doc_encoder.embedding_dimension

    def encode_queries(self, data: EncodingModelBatch) -> torch.Tensor:
        """Encode and normalize queries.

        Args:
            data (EncodingModelBatch): The encoder batch.

        Returns:
            torch.Tensor: The output representation.
        """
        reps = self.dropout(self.query_encoder(data))
        if self.projection is not None:
            reps = self.projection(reps)
        return torch.nn.functional.normalize(reps)

    def encode_docs(self, data: EncodingModelBatch) -> torch.Tensor:
        """Encode and normalize documents.

        Args:
            data (EncodingModelBatch): The encoder batch.

        Returns:
            torch.Tensor: The output representation.
        """
        reps = self.dropout(self.doc_encoder(data))
        if self.projection is not None:
            reps = self.projection(reps)
        return torch.nn.functional.normalize(reps)

    def forward(
        self,
        inputs: Union[ModelBatch, EncodingModelBatch],
        action: str = "score",
    ) -> torch.Tensor:
        """Perform one of three actions:
            * "score": Compute query-document scores.
            * "encode_queries": Compute query representations.
            * "encode_docs": Compute document representations.

        Args:
            inputs (Union[ModelBatch, EncodingModelBatch]): Batch of corresponding inputs.
            action (str, optional): Action to perform. Defaults to "score".

        Returns:
            torch.Tensor: Corresponding outputs.
        """
        if action == "score":
            query_data, doc_data = inputs
            return (self.encode_queries(query_data) * self.encode_docs(doc_data)).sum(1)

        if action == "encode_queries":
            return self.encode_queries(inputs)

        if action == "encode_docs":
            return self.encode_docs(inputs)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> torch.Tensor:
        if self.training_mode != TrainingMode.CONTRASTIVE:
            return super().training_step(batch, batch_idx)
        (queries, pos_docs), (_, neg_docs), _ = batch

        queries_enc = self.encode_queries(queries)
        docs_enc = torch.cat((self.encode_docs(pos_docs), self.encode_docs(neg_docs)))
        scores = torch.exp(torch.matmul(queries_enc, docs_enc.T) * self.temperature)

        # the diagonal holds all positive scores due to the way the docs are concatenated
        loss = torch.mean(-torch.log(torch.diagonal(scores) / torch.sum(scores)))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: ValTestBatch, batch_idx: int) -> None:
        super().validation_step(batch, batch_idx)

        if self.visualize_embeddings or self.compute_kl_div:
            # use the same inputs (documents) for both encoders here
            (_, doc_inputs), _, _ = batch
            emb_d_enc = self(doc_inputs, action="encode_docs")
            emb_q_enc = self(doc_inputs, action="encode_queries")

            self.d_enc_embeddings.update(emb_d_enc)
            self.q_enc_embeddings.update(emb_q_enc)

        if self.compute_kl_div:
            self.kl_div.update(emb_d_enc, emb_q_enc)

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        if self.visualize_embeddings or self.compute_kl_div:
            emb_d_enc = self.d_enc_embeddings.compute()
            emb_q_enc = self.q_enc_embeddings.compute()

        if self.visualize_embeddings:
            self.logger.experiment.add_embedding(
                torch.vstack((emb_d_enc, emb_q_enc)),
                metadata=["d_enc"] * emb_d_enc.shape[0]
                + ["q_enc"] * emb_q_enc.shape[0],
                tag=f"epoch {self.trainer.current_epoch}",
                global_step=self.trainer.global_step,
            )

        if self.compute_kl_div:
            self.log(
                f"val_{self.kl_div.__class__.__name__}",
                self.kl_div.compute(),
                sync_dist=True,
            )

        self.d_enc_embeddings.reset()
        self.q_enc_embeddings.reset()
        self.kl_div.reset()
