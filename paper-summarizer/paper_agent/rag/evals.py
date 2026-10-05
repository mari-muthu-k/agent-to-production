"""A small retrieval eval (Day 2, section 6.7): hit rate @k and MRR on golden questions.

A question is a hit when one of the top-k chunks (no threshold) comes from the expected paper and
touches one of the expected pages. Retrieval is evaluated on its own, without the LLM: if the right
page never reaches the prompt, no prompt engineering can save the answer.
"""
import json
from dataclasses import dataclass, field
from functools import cache
from importlib import resources


@cache
def golden_questions(version: str = "v1") -> list:
    return json.loads(resources.files("paper_agent").joinpath(f"data/retrieval_golden_{version}.json").read_text())


@dataclass
class EvalReport:
    k: int
    rows: list = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return sum(r["hit"] for r in self.rows) / max(len(self.rows), 1)

    @property
    def mrr(self) -> float:
        return sum(1 / r["rank"] if r["rank"] else 0 for r in self.rows) / max(len(self.rows), 1)

    def show(self) -> None:
        for r in self.rows:
            mark = "✅" if r["hit"] else "❌"
            rank = r["rank"] or "-"
            print(f"{mark} rank {rank:>2}  {r['paper_id']:10} p.{r['expected_pages']}  {r['question'][:70]}")
        print(f"\nhit rate @{self.k}: {self.hit_rate:.0%}   MRR: {self.mrr:.2f}   ({len(self.rows)} questions)")


def is_hit(chunk, expected: dict) -> bool:
    return chunk.paper_id == expected["paper_id"] and bool(set(chunk.pages or [chunk.page]) & set(expected["pages"]))


def hit_rate_at_k(retriever, golden: list, k: int = 5, per_paper: bool = False) -> EvalReport:
    """per_paper=True searches only the expected paper (as when a user asks about one uploaded paper)."""
    report = EvalReport(k)
    retriever.embedder.embed([g["question"] for g in golden])     # one request for all questions (cached)
    for g in golden:
        hits = retriever.search(g["question"], k=k, paper_id=g["paper_id"] if per_paper else None)
        rank = next((i + 1 for i, h in enumerate(hits) if is_hit(h.chunk, g)), None)
        report.rows.append({"question": g["question"], "paper_id": g["paper_id"], "expected_pages": g["pages"],
                            "hit": rank is not None, "rank": rank})
    return report
