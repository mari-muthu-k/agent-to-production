"""Tiny deterministic text utilities: a 'tokenizer' for usage counts, truncation and stop sequences.

Not a real tokenizer. One piece is a word or a punctuation mark, together with the whitespace
before it, so "".join(pieces(text)) == text. Counts are close to what a real BPE gives for English.
"""
import re

_PIECE = re.compile(r"\s*\w+|\s*[^\w\s]|\s+")
_WORD = re.compile(r"[a-z0-9@]+(?:[-.'][a-z0-9]+)*")

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my no not of on or our so than that the their them then there these they this to too us was we were
what when where which who why will with would you your about after all also any just more most only other
over such very paper section sections use used using says say tell explain one does
""".split())


def pieces(text: str) -> list:
    return _PIECE.findall(text or "")


def count(text: str) -> int:
    return len(pieces(text))


def truncate(text: str, max_tokens: int) -> tuple:
    """Return (text, truncated?) keeping at most max_tokens pieces."""
    p = pieces(text)
    if len(p) <= max_tokens:
        return text, False
    return "".join(p[:max_tokens]), True


def apply_stop(text: str, stop) -> str:
    if not stop:
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = min((i for i in (text.find(s) for s in stops if s) if i != -1), default=-1)
    return text if cut == -1 else text[:cut]


def words(text: str) -> list:
    return [w[:-2] if w.endswith("'s") else w for w in _WORD.findall((text or "").lower().replace("’", "'"))]


def content_words(text: str) -> set:
    return {w.rstrip("s") if len(w) > 4 else w for w in words(text) if w not in STOPWORDS and len(w) > 2}


def sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", (text or "").strip())
    return [s.strip() for s in parts if s.strip()]
