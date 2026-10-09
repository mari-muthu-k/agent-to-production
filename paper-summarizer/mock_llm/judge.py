"""Judge mode (Day 4, 7.2): what mock-llm does when the system prompt asks for a {"faithful": ...} verdict.

Like a strict reader: every sentence of the answer must be backed by the sources. A sentence counts as backed
when most of its content words appear in them; the agent's answers are built from the sources, so only a
planted claim ("trained on 10 trillion tokens of Rust") is unsupported. A refusal is faithful.
"""
import json
import re

from mock_llm.responders import Reply
from mock_llm.text import content_words, sentences

_SPLIT = re.compile(r"Sources:\s*(?P<sources>.*?)\n\s*Answer:\s*(?P<answer>.*)\Z", re.S)
_MARKER = re.compile(r"\s*\[[A-Za-z0-9][\w.:-]{0,39}\]")
_REFUSAL = re.compile(r"\b(doesn'?t|does not) cover\b", re.I)


def is_judge_request(req) -> bool:
    return '"faithful"' in req.system and not req.tools


def unsupported(answer: str, sources: str) -> list:
    known = content_words(sources)
    out = []
    for sentence in sentences(_MARKER.sub("", answer)):
        words = content_words(sentence)
        if words and len(words & known) / len(words) < 0.6:
            out.append(sentence)
    return out


def respond(req) -> Reply:
    m = _SPLIT.search(req.last_user)
    if not m:
        return Reply(content=json.dumps({"faithful": False, "reason": "no sources or no answer to compare"}))
    answer, sources = m["answer"].strip(), m["sources"]
    if _REFUSAL.search(answer):
        return Reply(content=json.dumps({"faithful": True, "reason": "a refusal makes no claims"}))
    bad = unsupported(answer, sources)
    reason = f"not in the sources: {bad[0][:80]!r}" if bad else "every claim is stated in the sources"
    return Reply(content=json.dumps({"faithful": not bad, "reason": reason}))
