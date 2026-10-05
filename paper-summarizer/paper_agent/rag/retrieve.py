"""Retrieval with a score threshold (Day 2, section 6.1-6.2).

`search` returns the k nearest chunks no matter how weak; `retrieve` keeps only chunks scoring at
least `min_score`. An empty result is the "insufficient evidence" signal: better to say so than to
hand the model k irrelevant chunks and invite it to improvise.

The right threshold depends on the embedding model (scores are not comparable across models), so
calibrate it: look at scores for questions the paper answers and for ones it does not (6.2).
"""
from dataclasses import dataclass
from typing import Optional


def keep_confident(hits: list, min_score: float) -> list:
    """Drop hits whose similarity is below the threshold."""
    return [h for h in hits if h.score >= min_score]


@dataclass
class Retriever:
    store: object
    embedder: object
    k: int = 5
    min_score: float = 0.25

    def search(self, question: str, k: Optional[int] = None, paper_id: Optional[str] = None) -> list:
        query = self.embedder.embed([question])[0]
        return self.store.search(query, k or self.k, paper_id=paper_id, embed_model=self.embedder.model)

    def retrieve(self, question: str, k: Optional[int] = None, paper_id: Optional[str] = None,
                 min_score: Optional[float] = None) -> list:
        threshold = self.min_score if min_score is None else min_score
        return keep_confident(self.search(question, k, paper_id), threshold)


def show_hits(hits: list, width: int = 90) -> None:
    for h in hits:
        c = h.chunk
        print(f"  {h.score:5.2f}  {c.paper_id}  {c.section[:28]:28} p.{c.page:<3} {c.text[:width]!r}")
