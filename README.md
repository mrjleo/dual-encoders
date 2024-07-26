# Dual-Encoders for IR

This repository implements dual-encoder models for information retrieval. Specifically, the following is supported:

- Training of Transformer-based symmetric or asymmetric dual-encoders
- Indexing corpora from [`ir_datasets`](https://ir-datasets.com/)
- Re-ranking using [Fast-Forward indexes](https://github.com/mrjleo/fast-forward-indexes)
- Dense retrieval using [FAISS](https://github.com/facebookresearch/faiss)

## Installation

First clone this repository locally and cd into it:
```
git clone git@github.com:BovdBerg/dual-encoders.git
cd dual-encoders/
```

Then create a new conda environment with python 3.8 and install the packages from `requirements.txt` (Note that [ranking_utils](https://github.com/mrjleo/ranking-utils) has additional dependencies):
```
conda create -y -n dual-encoders python==3.9 pip
conda activate dual-encoders
pip install --user -r requirements.txt
```

## Usage

We use [Hydra](https://hydra.cc/) for the configuration of all scripts, such as models, hyperparameters, paths and so on. Please refer to the documentation for instructions how to use Hydra.

The default behavior of Hydra is to create a new directory, `outputs`, in the current working directory. In order to use a custom output directory, override the `hydra.run.dir` argument.

Currently, the following encoder models are available along with the corresponding configuration files:

- Standard Transformer (`config/ranker/encoder/transformer.yaml`)
- Embedding-based Transformer (`config/ranker/encoder/transformer_embedding.yaml`)
- Selective Transformer (`config/ranker/encoder/selective_transformer.yaml`)

More information about these models can be found in [our paper](https://dl.acm.org/doi/10.1145/3631939).

### Pre-Processing

Currently, datasets must be pre-processed in order to use them for training. Refer to [this guide](https://github.com/mrjleo/ranking-utils#dataset-pre-processing) for a list of supported datasets and pre-processing instructions.

### Training

We use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training. Use the training script to train a new model and save checkpoints. At least the following options must be set: `training_data.data_dir` and `training_data.fold_name`.

For example, in order to train a symmetric dual-encoder, run the following:

```
python train.py \
    ranker/encoder@ranker.query_encoder=transformer \
    ranker.doc_encoder.pretrained_model=bert-base-uncased \
    ranker/encoder@ranker.doc_encoder=transformer \
    ranker.doc_encoder.pretrained_model=bert-base-uncased \
    training_data.data_dir=/path/to/preprocessed/files \
    training_data.fold_name=fold_0
```

The default configuration for training can be found in `config/trainer/fine_tuning.yaml`. All defaults can be overriden via the command line.

You can further override or add new arguments to other components such as the [`pytorch_lightning.Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags). Some examples are:

- `+trainer.val_check_interval=5000` to validate every 5000 batches.
- `+trainer.limit_val_batches=1000` to use only 1000 batches of the validation data.

**Important**: Training using the DDP strategy (`trainer.strategy=ddp`) may throw an error due to unused parameters. This can be worked around using `trainer.strategy=ddp_find_unused_parameters_true`.

#### Model Parameters

Model (hyper)parameters can be overidden like any other argument. They are prefixed by `ranker.model` and `ranker.model.hparams`, respectively. Check `config/ranker/dual_encoder.yaml` for available options.

#### Validation

Validation using ranking metrics is enabled by default (see [ranking-utils](https://github.com/mrjleo/ranking-utils?tab=readme-ov-file#validation) for more information). This can be configured using the Trainer options (see `config/trainer/fine_tuning.yaml`).

### Indexing

Indexing is handled by the `index.py` script, and the corresponding configuration file can be found at `config/indexing.yaml`. Most importantly, the type of index to be created can be controlled using `index_writer`:

- `index_writer=fast_forward` creates a [Fast-Forward OnDiskIndex](https://mrjleo.github.io/fast-forward-indexes/docs/v0.2.0/fast_forward/index/disk.html#OnDiskIndex)
- `index_writer=faiss` creates a [FAISS IndexFlatIP](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

Furthermore, the document collection to be indexed is determined using `encoding_data` and the corresponding configuations:

- `encoding_data=h5_corpus` indexes the corpus from a [pre-processed dataset](#pre-processing) (`config/encoding_data/h5_corpus.yaml`)
- `encoding_data=irds_corpus` indexes a corpus from [`ir_datasets`](https://ir-datasets.com/) (`config/encoding_data/irds_corpus.yaml`)

#### Examples

Creating a Fast-Forward index from a pre-processed dataset corpus:

```
python index.py \
    encoding_data=h5_corpus \
    encoding_data.data_dir=/path/to/preprocessed/files \
    ckpt_path=/path/to/model/checkpoint.ckpt \
    data_loader.batch_size=512 \
    index_writer=fast_forward
```

Creating a FAISS index from an [`ir_datasets` corpus](https://ir-datasets.com/beir.html#beir/fever), indexing both `title` and `text` attributes of each document:

```
python index.py \
    encoding_data=irds_corpus \
    encoding_data.dataset=beir/fever \
    encoding_data.content_attributes="[title, text]" \
    ckpt_path=/path/to/model/checkpoint.ckpt \
    data_loader.batch_size=512 \
    index_writer=faiss
```

**Important**: The document encoder (`doc_encoder`) configuration (i.e., hyperparameters) must match the training stage, otherwise the checkpoint cannot be loaded.

### Retrieve-and-rerank

#### First-Stage (Sparse) Retrieval
Create a sparse run where you retrieve the top-k ranked docs for each query in the testset. 
The configuration can be found in `config/sparse_retrieval.yaml`. Note that each of these parameters can be overwritten.

Run the sparse retrieval as follows:
```
python /path/to/retrieve_sparse.py
```

#### Re-Ranking
The `re_rank.py` script performs interpolation-based re-ranking using the [Fast-Forward indexes](https://github.com/mrjleo/fast-forward-indexes) pipeline. The configuration can be found in `config/indexing.yaml`. This requries an existing Fast-Forward index (see [indexing](#indexing)) and a corresponding first-stage (sparse retrieval) run to be re-ranked. For example:

```
python re_rank.py \
    ckpt_path=/path/to/model/checkpoint.ckpt \
    dataset=msmarco-passage/trec-dl-2019/judged \
    index_dir=/path/to/fast_forward/index \
    name=my-model \
    metrics="[nDCG@10]" \
    alpha="[0.7]" \
    sparse_runfile=/path/to/trec/run.tsv \
    cutoff_sparse=5000
```

Here, the queries (and QRels) are taken from [`ir_datasets`](https://ir-datasets.com/msmarco-passage.html#msmarco-passage/trec-dl-2019/judged). If any metrics are provided using the `metrics` config option, they are parsed and computed using [`ir-measures`](https://ir-measur.es/). The sparse runfile (`sparse_runfile`) must be in standard TREC format.

### Dense Retrieval

Alternatively to Re-ranking above, the trained models can be used for dense retrieval using the `retrieve.py` script (configured using `config/retrieval.yaml`). This requries a FAISS index (see [indexing](#indexing)). For example:

```
python retrieve.py \
    ckpt_path=/path/to/model/checkpoint.ckpt \
    dataset=msmarco-passage/trec-dl-2019/judged \
    index_dir=/path/to/faiss/index \
    name=my-model \
    metrics=[nDCG@10]
```

**Important**: The query encoder (`query_encoder`) configuration (i.e., hyperparameters) must match the training stage, otherwise the checkpoint cannot be loaded.
