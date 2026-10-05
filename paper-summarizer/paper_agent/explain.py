"""The Day 1 explainer pipeline: prompt -> structured call -> citation guard (sections 5.4-5.7, 6.10c).

Same behaviour as the notebook; the notebook's globals (PAPER, SECTION_IDS, llm) are now
parameters, defaulting to the TinyCoder fixture.
"""
import logging
from typing import Optional

from paper_agent.fixtures.tinycoder import PAPER as TINYCODER
from paper_agent.llm_client import LLMClient
from paper_agent.prompts import load_prompt
from paper_agent.schemas import PaperExplainer

logger = logging.getLogger("paper_agent.explain")

EXPLAINER_PROMPT = load_prompt("explainer", "v1")


def paper_messages(section_ids=None, paper: Optional[dict] = None, prompt: str = EXPLAINER_PROMPT) -> list:
    paper = paper or TINYCODER
    section_ids = section_ids or list(paper["sections"])
    blocks = "\n".join(f'<section id="{sid}">{paper["sections"][sid]}</section>' for sid in section_ids)
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": f"Paper title: {paper['title']}\n{blocks}\n\nExplain this paper."}]


class CitationError(Exception):
    pass


def check_citations(result, allowed_ids: list) -> None:
    unknown = set(result.citations) - set(allowed_ids)
    if unknown:
        raise CitationError(f"cites sections that were not provided: {sorted(unknown)}")


def explain_paper(llm: LLMClient, section_ids=None, max_tokens: int = 900,
                  paper: Optional[dict] = None) -> PaperExplainer:
    paper = paper or TINYCODER
    section_ids = section_ids or list(paper["sections"])
    messages = paper_messages(section_ids, paper)
    result = llm.chat_structured(messages, PaperExplainer, max_tokens=max_tokens)
    try:
        check_citations(result, section_ids)
        return result
    except CitationError as e:
        logger.warning(f"citation check failed, repairing once: {e}")
        repair = messages + [
            llm.last_message or {"role": "assistant", "content": result.model_dump_json()},   # keeps reasoning_details
            {"role": "user", "content": f"{e}. Cite only these ids: {section_ids}. Reply with ONLY the corrected JSON."},
        ]
        result = llm.chat_structured(repair, PaperExplainer, max_tokens=max_tokens)
        check_citations(result, section_ids)     # still wrong? fail loudly
        return result


def show(e: PaperExplainer):
    print(f"📰 {e.headline}   [{e.evidence_type}]  cites {e.citations}")
    print(e.summary)
    print("Why it matters:", e.why_it_matters)
    for t in e.key_terms: print(f"  • {t.term}: {t.plain_definition}")
    for c in e.caveats:   print(f"  ⚠ {c}")
