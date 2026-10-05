"""Deterministic offline embeddings for /v1/embeddings (see paper_agent/ingest/hashing.py).

The model name is the hashing salt, so vectors from two different embedding models are unrelated,
as with real providers. Use the same EMBED_MODEL for indexing and querying.
"""
import base64

import numpy as np

from paper_agent.ingest.hashing import hashing_embed

DEFAULT_DIM = 256


def embed(text: str, dim: int = DEFAULT_DIM, model: str = "") -> np.ndarray:
    return hashing_embed(text, dim, salt=model)


def encode(vec: np.ndarray, encoding_format: str) -> object:
    if encoding_format == "base64":
        return base64.b64encode(vec.astype("<f4").tobytes()).decode()
    return [float(x) for x in vec]
