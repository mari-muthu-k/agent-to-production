"""Agent mode (Day 3): what mock-llm does when it is offered the summarizer's tools.

A small scripted policy, keyed on the question and on the conversation so far, so every Day 3 cell is
deterministic offline and reads like a real model's trace:

  1. tool choice from descriptions only (paper_agent/agent/policy.py): get_section / define_term /
     save_note are picked only if their description says when to use them; otherwise search_paper
  2. multi-step plans for the questions the notebook asks (4.1: main result -> long files -> limitations;
     summaries: main result -> discussion -> limitations), and two get_section calls in ONE reply when a
     question names two sections (3.3)
  3. runaway: while a tool says "results incomplete", search again with a new query (4.5)
  4. injection: in "obey" mode (the default), instructions hidden in a tool result that is NOT inside
     <untrusted_document_text> tags are obeyed: the summary drops its caveats and save_note is called.
     In "resist" mode, or once the guard has wrapped the text, they are ignored (5.4-5.8)
  5. after the tools: an answer built from the returned chunks, citing their ids; "force_citation" adds
     a citation of a chunk that was never retrieved (5.7)

Switch modes with headers (X-Mock-Injection: obey|resist, X-Mock-Force-Citation: c99) or POST /mock/mode.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from mock_llm.responders import Reply, ToolCall, text_of
from mock_llm.text import content_words, sentences
from paper_agent.agent.policy import covers, is_profile, note_from, pick_tool, sections_named, term_asked

AGENT_TOOLS = {"search_paper", "get_section", "define_term", "save_note"}
_CHUNK = re.compile(r'<chunk id="([^"]+)" section="([^"]*)" page="(\d+)">\s*(.*?)\s*</chunk>', re.S)
_WRAPPED = re.compile(r"<untrusted_document_text\b[^>]*>.*?</untrusted_document_text>", re.S)
_INJECTED = re.compile(r"\b(ai|llm|language model)s?( tools?| assistants?| models?)?\b[^.\]]{0,60}\b(must|should)\b"
                       r"|\bassistant\s*:\s*save a note|ignore (all |any )?(previous|prior|above) instructions", re.I)
_EMAIL = re.compile(r"\S+@\S+\.[a-z]{2,}|\[REDACTED_EMAIL\]")
_SAVE_ORDER = re.compile(r"assistant\s*:\s*save a note (?:that )?(?P<note>[^.]+)", re.I)
_FOLLOW_UP = re.compile(r"\b(it|that|this)\b.*\b(again|simpl\w*|more|like i'?m|new to)\b|"
                        r"\b(explain|say) (it|that|this)\b", re.I)
RUNAWAY_QUERIES = ["evaluation benchmark", "how was TinyCoder evaluated", "evaluation results and metrics",
                   "test set and baseline", "evaluation details", "benchmark functions and unit tests"]
FOCUS = [  # extra words that make a sentence a good answer to a kind of question
    (r"weaken|main claim", "41% first attempt pass@1 42% baseline longer 1,024 drops 31% 39%"),
    (r"summar|overview|explain this paper", "present tinycoder 1.3-billion baseline faster 4× within point"),
    (r"main result", "41% first attempt pass@1 42% baseline faster"),
]


@dataclass
class Mode:
    injection: str = "obey"                  # obey | resist
    force_citation: Optional[str] = None     # e.g. "c99"


@dataclass
class Chunk:
    id: str
    section: str
    page: int
    text: str


@dataclass
class Run:
    """The current question and what the agent has done for it so far."""
    question: str
    earlier_questions: list
    calls: list = field(default_factory=list)       # (name, args) in order
    results: list = field(default_factory=list)     # (tool name, content) in order

    @property
    def last_result(self) -> str:
        return self.results[-1][1] if self.results else ""

    def chunks(self) -> list:
        found, seen = [], set()
        for _, content in self.results:
            for cid, section, page, body in _CHUNK.findall(content):
                if cid not in seen:
                    seen.add(cid)
                    found.append(Chunk(cid, section, int(page), body))
        return found

    def unwrapped_text(self) -> str:
        """Tool output outside <untrusted_document_text> tags: what the model reads as plain text."""
        return "\n".join(_WRAPPED.sub("", content) for _, content in self.results)

    def all_text(self) -> str:
        return "\n".join(content for _, content in self.results)


def is_agent_request(req) -> bool:
    names = {(t.get("function") or {}).get("name") for t in (req.tools or [])}
    return bool(names & AGENT_TOOLS) and not isinstance(req.tool_choice, dict) and req.tool_choice != "none"


def read_run(req) -> Run:
    messages = req.messages
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    run = Run(text_of(messages[last_user]) if last_user >= 0 else "",
              [text_of(m) for m in messages[:max(last_user, 0)] if m.get("role") == "user"])
    names = {}
    for m in messages[last_user + 1:]:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            names[c.get("id")] = fn.get("name")
            run.calls.append((fn.get("name"), json.loads(fn.get("arguments") or "{}")))
        if m.get("role") == "tool":
            run.results.append((m.get("name") or names.get(m.get("tool_call_id"), "tool"), text_of(m)))
    return run


def plan(run: Run, tools: dict) -> list:
    """The tool calls this question needs: a list of steps, each a list of (name, args) sent in one reply."""
    q = run.question
    lower = q.lower()
    search = "search_paper" if "search_paper" in tools else next(iter(tools))

    def section(sid: str):
        return ("get_section", {"section_id": sid}) if "get_section" in tools else (search, {"query": sid})

    if is_profile(q) and "save_note" in tools and covers(tools["save_note"].get("description"), "note"):
        return [[("save_note", {"note": note_from(q)})]]
    if _FOLLOW_UP.search(q) and not sections_named(q) and not term_asked(q):
        return [[(search, {"query": run.earlier_questions[-1]})]] if run.earlier_questions else []
    named = sections_named(q)
    if len(named) >= 2 and "get_section" in tools and covers(tools["get_section"].get("description"), "section"):
        return [[section(s) for s in named[:2]]]                      # two calls in one reply (3.3)
    if re.search(r"weaken|main claim", lower):
        return [[(search, {"query": "TinyCoder main result pass@1 compared with the 7B baseline"})],
                [(search, {"query": "files longer than 1,024 tokens"})], [section("limitations")]]
    if re.search(r"summar|overview|explain this paper", lower) and not named:
        return [[(search, {"query": "TinyCoder main result"})], [section("discussion")], [section("limitations")]]
    if re.search(r"who wrote|contact|authors", lower):
        return [[(search, {"query": "authors contact email"})]]
    return [[pick_tool(q, list(tools.values()))]]


def next_calls(run: Run, tools: dict, obey: bool) -> list:
    done = [(n, json.dumps(a, sort_keys=True)) for n, a in run.calls]
    order = _SAVE_ORDER.search(run.unwrapped_text())
    if obey and order and "save_note" in tools and not any(n == "save_note" for n, _ in run.calls):
        return [("save_note", {"note": order["note"].strip().capitalize() + "."})]        # memory poisoning
    if "results incomplete" in run.last_result:                                          # runaway (4.5)
        n = sum(1 for name, _ in run.calls if name == "search_paper")
        query = RUNAWAY_QUERIES[n % len(RUNAWAY_QUERIES)] + (f" (attempt {n + 1})" if n >= len(RUNAWAY_QUERIES) else "")
        return [("search_paper", {"query": query})]
    m = re.search(r"not found\. Valid section ids: (.+)\.$", run.last_result)
    if m and "get_section" in tools:                                                     # recover (3.5)
        valid = [s.strip() for s in m.group(1).split(",")]
        words = content_words(run.question)
        best = max(valid, key=lambda s: len(words & content_words(s.replace("-", " "))))
        if ("get_section", json.dumps({"section_id": best}, sort_keys=True)) not in done:
            return [("get_section", {"section_id": best})]
    for step in plan(run, tools):
        todo = [(n, a) for n, a in step if (n, json.dumps(a, sort_keys=True)) not in done]
        if todo:
            return todo
    return []


def evidence(chunk: Chunk) -> list:
    """The sentences of a chunk that can support an answer: no headings, no table rows, no injected text."""
    def prose(line: str) -> bool:
        words = line.split()
        numeric = sum(any(ch.isdigit() for ch in w) for w in words)
        return (len(words) >= 8 and not re.fullmatch(r"(\d+(\.\d+)*\s+)?[A-Z][\w ,:&()'/-]{2,60}", line.strip())
                and not line.startswith("Table ") and numeric <= len(words) // 3)
    lines = [ln for ln in chunk.text.splitlines() if prose(ln)]
    out = []
    for s in sentences(" ".join(" ".join(lines).split())):
        if s.endswith((".", "?", "!")) and len(s) > 30 and not _INJECTED.search(s) and not _EMAIL.search(s):
            out.append(s)
    return out


def compose(run: Run, req, mode: Mode, obey: bool) -> str:
    """The final answer: sentences from the returned chunks, each followed by its chunk id."""
    q, chunks = run.question, run.chunks()
    short = bool(re.search(r"reader profile.*\b(short|brief|concise)", req.system, re.I | re.S))
    if any(name == "save_note" and content.startswith("Saved") for name, content in run.results) and is_profile(q):
        return f"Noted: {note_from(q)} I'll keep that in mind in future answers."
    if not run.results:
        if _FOLLOW_UP.search(q):
            return ('I\'m not sure what "it" refers to: I don\'t have an earlier conversation with you in this '
                    "thread. Which term or section of the paper do you mean?")
        return "Ask me anything about the paper."
    if not chunks:
        if any(c == "not defined in this paper" for _, c in run.results):
            return f"The paper doesn't define {term_asked(q) or 'that term'}: it isn't defined in this paper."
        first = sentences(_WRAPPED.sub("", run.last_result) or run.last_result)[:2]
        return "Based on the tool result: " + (" ".join(first) if first else run.last_result[:300])
    if obey:
        cited = chunks[0].id
        return (f"TinyCoder is better than all large models at Python code completion [{cited}]. "
                f"It is small and fast [{cited}].") + _forced(mode)
    if re.search(r"who wrote|contact|authors", q, re.I):
        for c in chunks:
            who = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*,? and [A-Z][a-z]+ [A-Z][a-z]+) "
                            r"\(fictional authors\)", c.text)
            how = re.search(r"Contact[^:\n]*:\s*((?:(?:[\w.+-]+@[\w.-]+\.[a-z]{2,}|\[REDACTED_EMAIL\])(?:,\s*)?)+)",
                            c.text)
            if who and how:
                return (f"The authors are {who.group(1).strip()} (fictional) [{c.id}]. You can contact them by email: "
                        f"{how.group(1).strip()} [{c.id}].") + _forced(mode)
    term = term_asked(q)
    if term and any(name == "define_term" for name, _ in run.calls):
        sentence = (evidence(chunks[0]) or [chunks[0].text])[0]
        return f"In this paper, {term} is defined like this: {sentence} [{chunks[0].id}]" + _forced(mode)
    focus = " ".join(words for pattern, words in FOCUS if re.search(pattern, q, re.I))
    wanted = content_words(q + " " + focus) | {w for w in focus.split() if not w.isalpha()}
    pool = [(c, s) for c in chunks for s in evidence(c)]
    overlap = [len(wanted & (content_words(s) | set(s.split()))) for _, s in pool]
    scored = sorted(range(len(pool)), key=lambda i: (-overlap[i], i))
    picked = sorted(scored[:1 if short else 3])
    lines = [pool[i] for i in picked]
    if re.search(r"weaken|main claim|summar|overview|explain this paper", q, re.I) and not short:
        limits = [(c, s) for c in chunks if "imitation" in c.section or "imitations" in c.text[:40]
                  for s in evidence(c) if re.search(r"\bonly\b|not evaluated|one benchmark", s, re.I)]
        if limits and limits[0] not in lines:
            lines.append(limits[0])
    plain = "In plain words: " if re.search(r"new to|simpl", q, re.I) else ""
    answer = plain + " ".join(f"{s.rstrip('.')} [{c.id}]." for c, s in lines)
    if _INJECTED.search(run.all_text()):
        answer += (" (The document contains hidden instructions aimed at AI tools; I treated them as data and "
                   "ignored them.)")
    return answer + _forced(mode)


def _forced(mode: Mode) -> str:
    return f" See also [{mode.force_citation}]." if mode.force_citation else ""


def respond(req, fault, mode: Mode) -> Reply:
    tools = {(t.get("function") or {}).get("name"): t["function"] for t in req.tools or [] if t.get("function")}
    run = read_run(req)
    hidden = _INJECTED.search(run.unwrapped_text())
    obey = bool(hidden) and (mode.injection == "obey" or fault.obey_injection)
    calls = next_calls(run, tools, obey) if req.tool_choice != "none" else []
    if calls:
        n = sum(len(m.get("tool_calls") or []) for m in req.messages)
        return Reply(tool_calls=[ToolCall(name, args, f"call_{n + i}") for i, (name, args) in enumerate(calls)])
    return Reply(content=compose(run, req, mode, obey))
