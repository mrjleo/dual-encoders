import logging
from pathlib import Path
from typing import Union

import numpy as np
from fast_forward.encoder import QueryEncoder as FFQueryEncoder
from fast_forward.index import InMemoryIndex, Mode

from util import read_faiss_index

LOGGER = logging.getLogger(__name__)


def faiss_to_ff(
    index_dir: Union[str, Path],
    query_encoder: FFQueryEncoder = None,
    mode: Mode = Mode.MAXP,
    encoder_batch_size: int = 32,
) -> InMemoryIndex:
    """Read a FAISS index from disk and create a Fast-Forward index using the vectors and IDs.

    Args:
        index_dir (Union[str, Path]): Directory where the FAISS index is stored.
        query_encoder (FFQueryEncoder, optional): A query encoder for the FF index. Defaults to None.
        mode (Mode, optional): FF index mode. Defaults to Mode.MAXP.
        encoder_batch_size (int, optional): FF query encoder batch size. Defaults to 32.

    Returns:
        InMemoryIndex: The FF index.
    """
    index_dir = Path(index_dir)
    LOGGER.info("reading %s", index_dir)
    faiss_index, orig_doc_ids = read_faiss_index(index_dir)
    faiss_ids, orig_ids = zip(*orig_doc_ids.items())
    faiss_ids = np.array(faiss_ids)
    orig_ids = list(orig_ids)

    LOGGER.info("reconstructing vectors")
    # the vectors are not necessarily in the same order as the ids, hence we need to permute
    vectors = faiss_index.reconstruct_n(0, faiss_ids.shape[0])[faiss_ids]

    LOGGER.info("creating FF index")
    ff_index = InMemoryIndex(query_encoder, mode, encoder_batch_size)
    ff_index.add(vectors, doc_ids=orig_ids)
    return ff_index


def read_ff_index(
    index_dir: Union[str, Path],
    query_encoder: FFQueryEncoder = None,
    mode: Mode = Mode.MAXP,
    encoder_batch_size: int = 32,
) -> InMemoryIndex:
    """Read a Fast-Forward index from disk.

    Args:
        index_dir (Union[str, Path]): Directory where the Fast-Forward index is stored.
        query_encoder (FFQueryEncoder, optional): A query encoder for the FF index. Defaults to None.
        mode (Mode, optional): FF index mode. Defaults to Mode.MAXP.
        encoder_batch_size (int, optional): FF query encoder batch size. Defaults to 32.

    Returns:
        InMemoryIndex: The FF index.
    """
    index_dir = Path(index_dir)
    LOGGER.info("reading %s", index_dir)
    return InMemoryIndex.from_disk(
        index_dir / "index.pkl", query_encoder, mode, encoder_batch_size
    )
