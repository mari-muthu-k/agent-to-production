"""A small semantic cache (Day 3, 6.4): numpy cosine over Day 2's embeddings, one cache per paper.

A reworded question ("What pass@1 does TinyCoder get?") can reuse a saved answer. So can a question that
only LOOKS the same: pass@1 and pass@10 differ by one character, and an embedding model sees them as
nearly identical. Keep semantic caching for FAQ-style questions, use a high threshold calibrated on
near-misses like that one, scope it to one paper, and never store an answer a guard rejected.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from paper_agent.rag import pdf_rag as day2


@dataclass
class Lookup:
    answer: Optional[str]          # None: a miss
    score: float                   # similarity to the closest cached question (-1 if the cache is empty)
    question: Optional[str]        # that cached question


class SemanticCache:
    def __init__(self, embed_fn, threshold: float, scope: str):
        self.embed_fn = embed_fn      # list of texts -> one vector per row, e.g. Day 2's embed_texts
        self.threshold = threshold
        self.scope = scope            # the paper this cache serves: answers about another paper never match
        self.questions: list = []
        self.answers: list = []
        self.vectors = None

    def lookup(self, question: str) -> Lookup:
        if not self.questions:
            return Lookup(None, -1.0, None)
        scores = day2.cosine_scores(self.vectors, np.asarray(self.embed_fn([question])[0], dtype=np.float32))
        best = int(np.argmax(scores))
        score = float(scores[best])
        return Lookup(self.answers[best] if score >= self.threshold else None, score, self.questions[best])

    def put(self, question: str, answer: str, passed_guards: bool) -> bool:
        """Store an answer, unless a guard failed it (citation check, limits, budget): those are never cached."""
        if not passed_guards:
            return False
        vector = np.asarray(self.embed_fn([question]), dtype=np.float32)
        self.vectors = vector if self.vectors is None else np.vstack([self.vectors, vector])
        self.questions.append(question)
        self.answers.append(answer)
        return True
