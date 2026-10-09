"""Scorers (Day 4, 7.2): rules a program checks exactly, then an LLM judge for what rules can't check.

No textstat: its pinned version downloads NLTK data at runtime, which fails offline. reading_ease() is the
Flesch formula with a small syllable counter: rough, but the same rough on every run, which is what a
comparison with a baseline needs.
"""
import json
import re
from collections.abc import Callable
from typing import Optional

from paper_agent.agent.guards import CITATION
from paper_agent.prompts import load_prompt

JUDGE_VERSION = "judge_v1"
JUDGE_V1 = load_prompt("judge", "v1")
REFUSAL = "The paper doesn't cover that."
HYPE = ("breakthrough", "revolutionary", "game-changing", "groundbreaking", "unprecedented", "better than all",
        "state-of-the-art", "best ever", "incredible", "amazing")
REFUSED = re.compile(r"\b(doesn'?t|does not|didn'?t|did not) (cover|say|mention|report|state)|\bnot (covered|mentioned|"
                     r"reported|stated) in the paper", re.I)


def plain(text: str) -> str:
    """The answer without citation markers like [p2-c1]."""
    return " ".join(CITATION.sub("", text or "").split())


def syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and groups > 1:
        groups -= 1                                   # silent e: "make", "code"
    return max(1, groups)


def reading_ease(text: str) -> float:
    """Flesch reading ease, kept between 0 and 100: higher is easier. 60-70 is plain English; under 30 reads
    like a paper."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*|\d[\d,.%@]*", text)]
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r"[.!?]+(\s|$)", text)))
    syl = sum(syllables(w) if w[0].isalpha() else 1 for w in words)
    return round(min(100.0, max(0.0, 206.835 - 1.015 * len(words) / sentences - 84.6 * syl / len(words))), 1)


def is_refusal(answer: str) -> bool:
    return bool(REFUSED.search(answer or ""))


def score(item: dict, result: dict, chunks: dict, faithful: Optional[bool] = None) -> dict:
    """One question's scores. `result` is ask_paper()'s (or the api's) reply; `chunks` maps chunk id ->
    {"section": section id, "page": n}. None = this metric doesn't apply to this question."""
    answerable, answer, cited = item["answerable"], result["answer"], result.get("citations") or []
    sections = {chunks[c]["section"] for c in cited if c in chunks}
    pages = {chunks[c]["page"] for c in cited if c in chunks}
    retrieved_pages = {chunks[c]["page"] for c in result.get("retrieved") or [] if c in chunks}
    return {
        "valid_citations": 1.0 if result.get("citations_valid", True) else 0.0,
        "cited": (1.0 if cited else 0.0) if answerable else None,
        "right_section": (1.0 if sections & set(item["expected_sections"]) else 0.0) if answerable else None,
        "faithful": None if faithful is None else (1.0 if faithful else 0.0),
        "refused_correctly": None if answerable else (1.0 if is_refusal(answer) else 0.0),
        "hype_free": 0.0 if any(h in answer.lower() for h in HYPE) else 1.0,
        "readability": reading_ease(plain(answer)) if answerable and not is_refusal(answer) else None,
        "recall_at_k": (1.0 if item.get("expected_page") in retrieved_pages else 0.0) if answerable else None,
        "page_accuracy": (1.0 if item.get("expected_page") in pages else 0.0) if answerable else None,
    }


def judge_messages(question: str, sources: str, answer: str) -> list:
    return [{"role": "system", "content": JUDGE_V1},
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}\n\nAnswer:\n{answer}"}]


def parse_verdict(reply: str) -> dict:
    """{"faithful": bool, "reason": str}. Anything we can't read counts as NOT faithful: a judge that fails
    must never make a bad answer look good."""
    match = re.search(r"\{.*\}", reply or "", re.S)
    try:
        verdict = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("faithful"), bool):
        return {"faithful": False, "reason": f"unparseable judge reply: {(reply or '')[:60]!r}"}
    return {"faithful": verdict["faithful"], "reason": str(verdict.get("reason", ""))[:200]}


def judge(question: str, sources: str, answer: str, complete: Callable) -> dict:
    """A second model checks that every claim is supported. complete(messages, max_tokens) -> reply text."""
    from paper_agent.service.config import JUDGE_MAX_TOKENS
    return parse_verdict(complete(judge_messages(question, sources, answer), JUDGE_MAX_TOKENS))
