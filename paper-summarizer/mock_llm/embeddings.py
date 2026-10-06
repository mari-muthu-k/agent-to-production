"""Deterministic offline embeddings for /v1/embeddings (see paper_agent/ingest/hashing.py).

The model name is the hashing salt, so vectors from two different embedding models are unrelated,
as with real providers. Use the same EMBED_MODEL for indexing and querying.
"""
import base64
import re

import numpy as np

from paper_agent.ingest.hashing import hashing_embed

DEFAULT_DIM = 1024   # wide enough that two "models" (salts) score near zero against each other


# The hashing "model" only sees shared words. Real embedding models also match related words, so the mock
# folds a few word forms and synonyms together: just enough for the Day 2 narration to hold offline
# ("do on long files" finds "files longer than 1,024 tokens"; "learning rate" lands near the training text).
SYNONYMS = {"longer": "long", "learning": "training", "trained": "training", "train": "training",
            "checked": "check", "checking": "check", "handle": "", "handles": "", "conclude": "conclusion",
            "concludes": "conclusion"}
_WORD = re.compile(r"[A-Za-z]+")


def normalise(text: str) -> str:
    return _WORD.sub(lambda m: SYNONYMS.get(m.group(0).lower(), m.group(0)), text or "")


def embed(text: str, dim: int = DEFAULT_DIM, model: str = "") -> np.ndarray:
    return hashing_embed(normalise(text), dim, salt=model)


def encode(vec: np.ndarray, encoding_format: str) -> object:
    if encoding_format == "base64":
        return base64.b64encode(vec.astype("<f4").tobytes()).decode()
    return [float(x) for x in vec]
