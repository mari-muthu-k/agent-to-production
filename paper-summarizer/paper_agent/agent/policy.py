"""How the workshop's offline models choose a tool: from the tool descriptions alone (Day 3, 3.2 and 3.4).

Shared by mock-llm (mock_llm/agent.py) and the offline fake model (paper_agent/agent/fakes.py), so the
docstring demo behaves the same in the notebook, in the TODO tests and in pytest. Pure Python, no
LangChain: mock-llm imports it.

The rule imitates what real models do with a weak description: a specialised tool is only picked when
its description says WHEN to use it ("Use this when the reader names a section ..."); otherwise the
model falls back to the general search tool.
"""
import re

SECTION_WORDS = ("abstract", "introduction", "methods", "method", "results", "discussion", "limitations",
                 "conclusion", "conclusions", "evaluation", "references")
# What a "when to use" sentence must mention to cover each kind of question
INTENT_CUES = {
    "section": ("section",),
    "definition": ("mean", "defin", "term", "jargon"),
    "note": ("about themselves", "about the reader", "prefer", "profile", "remember"),
}
_WHEN = re.compile(r"\b(use (this|it|me)|call (this|it)|use when|use for)\b.*\b(when|whenever|if|for)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_DEFINE = re.compile(r"what does (?P<a>.+?) mean|what is meant by (?P<b>.+?)[?.]|(?:define|definition of) "
                     r"(?P<c>.+?)[?.]|meaning of (?P<d>.+?)[?.]", re.I)
_PROFILE = re.compile(r"\bI(?:'m| am) (?:an? )?[\w\- ]+?(?: and|,)? .*\b(prefer|like|want)\b", re.I)


def when_to_use(description: str) -> str:
    """The sentences of a tool description that say when to use the tool ("" if there are none)."""
    return " ".join(s for s in _SENTENCE.split(description or "") if _WHEN.search(s))


def covers(description: str, intent: str) -> bool:
    when = when_to_use(description).lower()
    return bool(when) and any(cue in when for cue in INTENT_CUES[intent])


def sections_named(question: str) -> list:
    """Section ids the question names, in order: 'the limitations section' -> ['limitations']."""
    q = question.lower()
    named = [w for w in SECTION_WORDS if re.search(rf"\b{w}\b", q)]
    return sorted(named, key=q.index) if "section" in q else []


def term_asked(question: str):
    """The term in 'What does pass@1 mean in this paper?', or None."""
    m = _DEFINE.search(question)
    if not m:
        return None
    term = next(g for g in m.groups() if g)
    term = re.sub(r"\s+in (this|the) paper$", "", term.strip(" '\"“”"), flags=re.I)
    return term.strip(" '\"“”") or None


def is_profile(question: str) -> bool:
    return bool(_PROFILE.search(question))


def intent(question: str) -> str:
    if is_profile(question):
        return "note"
    if term_asked(question):
        return "definition"
    if sections_named(question):
        return "section"
    return "search"


def pick_tool(question: str, tools: list):
    """(tool name, args) the offline model calls first for `question`.

    tools: OpenAI-format function dicts ({"name", "description", "parameters"}). Falls back to a
    tool whose name contains "search", then to the first tool.
    """
    kind = intent(question)
    by_name = {t["name"]: t for t in tools}
    if kind == "section" and "get_section" in by_name and covers(by_name["get_section"].get("description"), "section"):
        return "get_section", {"section_id": sections_named(question)[0]}
    if kind == "definition" and "define_term" in by_name and covers(by_name["define_term"].get("description"),
                                                                    "definition"):
        return "define_term", {"term": term_asked(question)}
    if kind == "note" and "save_note" in by_name and covers(by_name["save_note"].get("description"), "note"):
        return "save_note", {"note": note_from(question)}
    search = next((t["name"] for t in tools if "search" in t["name"]), tools[0]["name"])
    return search, {"query": question.strip()}


def note_from(statement: str) -> str:
    """'I'm a backend engineer and I prefer short, technical answers.' -> 'Backend engineer; prefers short,
    technical answers.'"""
    s = statement.strip().rstrip(".")
    m = re.match(r"I(?:'m| am) (?:an? )?(?P<who>[\w\- ]+?)(?: and|,) (?:I )?(?P<pref>prefer|like|want)s? (?P<what>.+)",
                 s, re.I)
    if not m:
        return s
    return f"{m['who'].strip().capitalize()}; {m['pref'].lower()}s {m['what'].strip()}."
