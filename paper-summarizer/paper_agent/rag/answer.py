"""Grounded answering with page citations (Day 2, section 6.3-6.6).

answer_question():
  1. retrieve chunks above the score threshold; none -> "insufficient evidence", no LLM call
  2. keep the best chunks that fit the context budget (no overflow, no burying the answer)
  3. ask for a GroundedAnswer: status, answer, (section, page) citations
  4. citation guard: every (section, page) must belong to a chunk we actually sent; repair once
  5. number check: numbers in the answer must appear in the cited chunks, else a warning
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from paper_agent.llm_client import LLMClient
from paper_agent.prompts import load_prompt
from paper_agent.schemas import GroundedAnswer
from paper_agent.tokens import count_tokens

logger = logging.getLogger("paper_agent.rag")

QA_PROMPT = load_prompt("qa", "v1")
NOT_FOUND = "The paper excerpts I retrieved do not answer this question."


class PageCitationError(Exception):
    pass


def fit_to_budget(hits: list, max_tokens: int) -> list:
    """Best-scoring hits first, stopping before the context budget is exceeded."""
    kept, used = [], 0
    for h in sorted(hits, key=lambda h: -h.score):
        n = count_tokens(h.chunk.text) + 20          # + tag overhead
        if used + n > max_tokens:
            break
        kept.append(h)
        used += n
    return kept


def rag_messages(question: str, hits: list, prompt: str = QA_PROMPT) -> list:
    context = "\n".join(h.chunk.tag() for h in hits)
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}]


def allowed_citations(hits: list) -> set:
    """Every (section, page) pair the model may cite: one per page each sent chunk touches."""
    return {(h.chunk.section, p) for h in hits for p in (h.chunk.pages or [h.chunk.page])}


def check_page_citations(result: GroundedAnswer, hits: list) -> None:
    """Every cited (section, page) must belong to a chunk we actually sent."""
    unknown = {(c.section, c.page) for c in result.citations} - allowed_citations(hits)
    if unknown:
        raise PageCitationError(f"cites (section, page) pairs that were not retrieved: {sorted(unknown)}")


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def unsupported_numbers(result: GroundedAnswer, hits: list) -> list:
    """Numbers in the answer that appear in none of the cited chunks (a cheap hallucination check)."""
    cited = {(c.section, c.page) for c in result.citations}
    text = " ".join(h.chunk.text for h in hits
                    if any((h.chunk.section, p) in cited for p in (h.chunk.pages or [h.chunk.page])))
    norm = lambda s: s.replace(",", "")  # noqa: E731
    have = {norm(n) for n in _NUMBER.findall(text)}
    return [n for n in _NUMBER.findall(result.answer) if norm(n) not in have]


@dataclass
class RAGAnswer:
    question: str
    status: str
    answer: str
    citations: list = field(default_factory=list)
    confidence: str = "low"
    hits: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    llm_called: bool = True

    def show(self) -> None:
        icon = "✅" if self.status == "answered" else "🤷"
        print(f"{icon} {self.question}\n   {self.answer}")
        if self.citations:
            print("   cites:", ", ".join(f"{c.section} p.{c.page}" for c in self.citations))
        for w in self.warnings:
            print("   ⚠", w)


def answer_question(question: str, retriever, llm: LLMClient, k: int = 5, min_score: Optional[float] = None,
                    paper_id: Optional[str] = None, max_context_tokens: int = 3000,
                    max_tokens: int = 800) -> RAGAnswer:
    hits = retriever.retrieve(question, k=k, paper_id=paper_id, min_score=min_score)
    if not hits:
        return RAGAnswer(question, "insufficient_evidence", NOT_FOUND, llm_called=False,
                         warnings=["no chunk scored above the threshold: answered without calling the model"])
    hits = fit_to_budget(hits, max_context_tokens)
    messages = rag_messages(question, hits)
    result = llm.chat_structured(messages, GroundedAnswer, max_tokens=max_tokens)
    try:
        check_page_citations(result, hits)
    except PageCitationError as e:
        logger.warning(f"citation check failed, repairing once: {e}")
        allowed = sorted(allowed_citations(hits))
        repair = messages + [
            llm.last_message or {"role": "assistant", "content": result.model_dump_json()},   # keeps reasoning_details
            {"role": "user", "content": f"{e}. Cite only these (section, page) pairs: {allowed}. "
                                        "Reply with ONLY the corrected JSON."}]
        result = llm.chat_structured(repair, GroundedAnswer, max_tokens=max_tokens)
        check_page_citations(result, hits)      # still wrong? fail loudly
    warnings = []
    if result.status == "answered" and (missing := unsupported_numbers(result, hits)):
        warnings.append(f"numbers not found in the cited pages: {missing}")
    return RAGAnswer(question, result.status, result.answer, result.citations, result.confidence, hits, warnings)
