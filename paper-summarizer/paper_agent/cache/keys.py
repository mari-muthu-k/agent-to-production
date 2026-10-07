"""Cache keys (Day 3, 6.3)."""
import hashlib
import json


def normalize(query: str) -> str:
    """Spelling-level variation only: case, spacing, trailing punctuation. Meaning stays exact."""
    return " ".join(query.lower().split()).strip(" ?!.")


def make_key(*parts) -> str:
    """One sha256 over every part, in order. A part that changes the answer and is missing here is a bug."""
    return hashlib.sha256(json.dumps([str(p) for p in parts]).encode()).hexdigest()
