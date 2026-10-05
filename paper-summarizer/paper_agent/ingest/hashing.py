"""Hashed bag-of-words embeddings in pure numpy (also what mock-llm serves for /v1/embeddings).

Words and word pairs are hashed into `dim` buckets with a sign, then L2-normalized. Texts that share
words get similar vectors. Synonyms do not ("car" vs "automobile" score 0), which is exactly the gap
real embedding models close.

`salt` stands in for "which model": the same text embedded with different salts lands in unrelated
places, just like vectors from two different embedding models.
"""
import hashlib
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9@]+(?:[-.'][a-z0-9]+)*")
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my no not of on or our so than that the their them then there these they this to too us was we were
what when where which who why will with would you your about after all also any just more most only other
over such very
""".split())


def tokens(text: str) -> list:
    out = []
    for w in _WORD.findall((text or "").lower().replace("’", "'")):
        w = w[:-2] if w.endswith("'s") else w
        if w in STOPWORDS or len(w) < 2:
            continue
        out.append(w.rstrip("s") if len(w) > 4 else w)   # crude plural folding: models -> model
    return out


def _bucket(feature: str, dim: int) -> tuple:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


def hashing_embed(text: str, dim: int = 256, salt: str = "") -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    toks = tokens(text)
    pairs = zip(toks, toks[1:], strict=False)
    feats = [(f"{salt}|w:{t}", 1.0) for t in toks] + [(f"{salt}|b:{a}_{b}", 0.5) for a, b in pairs]
    for feature, weight in feats:
        i, sign = _bucket(feature, dim)
        vec[i] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        vec[_bucket(f"{salt}|empty", dim)[0]] = 1.0
        return vec
    return vec / norm
