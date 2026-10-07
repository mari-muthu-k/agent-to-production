"""Tool-result cache (Day 3, 6.3): skip a repeated search on the same paper.

Search results depend on the paper, the embedding model, k and the query, so all four belong in the
key. A key with the query alone serves one paper's chunks for another paper: a wrong answer with
confident, correct-looking citations. If results were personal, the user id would go in the key too.
"""
import os

from paper_agent.agent.papers import PAPERS
from paper_agent.cache.keys import make_key, normalize
from paper_agent.rag import pdf_rag as day2

EMBED_MODEL = os.environ.get("EMBED_MODEL", "")
TOOL_CACHE: dict = {}


def naive_key(query: str, paper_hash: str, k: int) -> str:
    """The query text alone."""
    return normalize(query)


def search_key(query: str, paper_hash: str, k: int) -> str:
    """Everything that changes the search result."""
    key = make_key(paper_hash, EMBED_MODEL, k, normalize(query))
    return key


def cached_search(query: str, paper_id: str, key_fn, k: int = 4, cache=None, search_fn=None) -> tuple:
    """(hits, was_it_a_hit), with the cache key from key_fn(query, paper_hash, k). paper_hash is a hash of
    the PDF's content, not its file name: a re-uploaded paper with the same name but new content gets new keys."""
    cache = TOOL_CACHE if cache is None else cache
    key = key_fn(query, PAPERS[paper_id]["hash"], k)
    if key in cache:
        return cache[key], True
    hits = (search_fn or day2.retrieve)(query, paper_id=paper_id, k=k)
    cache[key] = hits
    return hits, False
