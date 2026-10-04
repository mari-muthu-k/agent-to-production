"""Deterministic offline embeddings: signed feature hashing of words and word pairs, L2-normalized.

Texts that share words get similar vectors, so retrieval demos work without a real model.
They do NOT capture meaning (synonyms score 0), which is itself a useful Day 2 teaching point.
"""
import base64
import hashlib

import numpy as np

from mock_llm.text import STOPWORDS, words

DEFAULT_DIM = 256


def _bucket(feature: str, dim: int) -> tuple:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


def embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    toks = [w for w in words(text) if w not in STOPWORDS and len(w) > 1]
    feats = [f"w:{w.rstrip('s') if len(w) > 4 else w}" for w in toks]
    feats += [f"b:{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
    for f in feats:
        i, sign = _bucket(f, dim)
        vec[i] += sign * (0.5 if f.startswith("b:") else 1.0)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        i, _ = _bucket("empty", dim)
        vec[i] = 1.0
        return vec
    return vec / norm


def encode(vec: np.ndarray, encoding_format: str) -> object:
    if encoding_format == "base64":
        return base64.b64encode(vec.astype("<f4").tobytes()).decode()
    return [float(x) for x in vec]
