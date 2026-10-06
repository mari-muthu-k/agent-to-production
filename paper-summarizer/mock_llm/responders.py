"""What the mock model 'says'. Deterministic for temperature 0, so notebooks and tests are reproducible.

Order of decisions for a chat request:
  1. tool_choice forces a function            -> a tool call whose arguments fit its JSON schema
  2. tools offered with tool_choice auto      -> call the best-matching tool, or answer after a tool result
  3. JSON requested (response_format / prompt) -> schema-valid JSON (PaperExplainer, Answer, or generic)
  4. a known teaching prompt                  -> a scripted reply (Day 1 sections 4-6, instructor demo)
  5. anything else                            -> a short extractive reply from the provided text

Prompt injection: the mock obeys instructions hidden in document text ONLY when the system prompt has
no "data, not instructions" rule (or X-Mock-Obey-Injection is set). That makes naive-vs-guarded demos
deterministic. Real models are less predictable: resisting once is not a guarantee.
"""
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from mock_llm.faults import Fault
from mock_llm.text import content_words, sentences

# ---------------------------------------------------------------------------
# Reading the request
# ---------------------------------------------------------------------------
_BLOCK = re.compile(r"<(section|chunk|source|document)\b([^>]*)>(.*?)</\1>", re.S | re.I)
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_INJECTION = re.compile(
    r"(\b(ai|llm|language model|assistant)s?\b[^.\]]{0,40}\b(must|should|are required to)\b"
    r"|ignore (all |any )?(previous|prior|above) instructions"
    r"|disregard (the )?(system|previous) (prompt|instructions))", re.I)
_IDK_RULES = ("insufficient_evidence", "insufficient evidence", "i don't know", "say you don't know",
              "not in the context", "if the excerpts do not", "if the chunks do not", "if the chunks don't",
              '"found" to false', "found to false", "found=false", "found = false")
_DEFENSES = ("never as instructions", "not as instructions", "never instructions", "untrusted",
             "do not follow instructions", "never follow instructions", "ignore instructions inside")


def text_of(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


@dataclass
class Block:
    id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class Request:
    messages: list
    model: str
    temperature: Optional[float] = None
    seed: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Any = None
    reasoning: bool = False          # OpenRouter `reasoning` switched on for this request

    @property
    def system(self) -> str:
        return "\n".join(text_of(m) for m in self.messages if m.get("role") in ("system", "developer"))

    @property
    def last_user(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return text_of(m)
        return ""

    @property
    def all_text(self) -> str:
        return "\n".join(text_of(m) for m in self.messages)

    @property
    def is_repair(self) -> bool:
        """A follow-up asking to fix a previous reply: the mock always gets it right the second time."""
        later = [text_of(m) for m in self.messages if m.get("role") == "user"][1:]
        return any(k in t for t in later for k in ("failed validation", "Cite only these", "corrected JSON"))

    def blocks(self) -> list:
        """Document blocks (<section id=..>, <chunk id=.. page=..>) from the user messages, in order."""
        found, seen = [], set()
        for m in self.messages:
            if m.get("role") not in ("user", "tool"):
                continue
            for _, attrs, body in _BLOCK.findall(text_of(m)):
                a = dict(_ATTR.findall(attrs))
                bid = a.get("id") or a.get("name") or f"block{len(found) + 1}"
                if bid in seen:
                    continue
                seen.add(bid)
                page = a.get("page")
                found.append(Block(bid, body.strip(), int(page) if page and page.isdigit() else None,
                                   a.get("section")))
        return found

    def title(self) -> str:
        m = re.search(r"(?:Paper )?title:\s*(.+)", self.all_text, re.I)
        return m.group(1).strip() if m else ""

    def rng(self) -> random.Random:
        temp = 1.0 if self.temperature is None else self.temperature
        if temp < 0.7:
            return random.Random(0)
        return random.Random(self.seed)   # seed=None -> fresh randomness, like a real model


def obeys_injection(req: Request, fault: Fault) -> bool:
    if fault.obey_injection:
        return True
    system = req.system.lower()
    return not any(d in system for d in _DEFENSES)


def injected_blocks(blocks: list) -> list:
    return [b for b in blocks if _INJECTION.search(b.text)]


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    """Drop bracketed hidden-text markers so they never leak into an answer."""
    return re.sub(r"\[[^\]]*hidden[^\]]*\]", "", text, flags=re.I).strip()


def _cite(blocks: list, preferred: list) -> list:
    ids = [b.id for b in blocks]
    chosen = [p for p in preferred if p in ids]
    return chosen or ids[:2] or ["abstract"]


TINYCODER_EXPLAINER = {
    "headline": "A small code model nearly matches a 7B model on one Python test",
    "summary": ("The authors built TinyCoder, a small AI model (1.3 billion parameters) that completes Python code. "
                "On one benchmark the authors built themselves, it solved 41% of tasks on the first try, "
                "versus 42% for a model five times bigger, and it ran about 4x faster. "
                "On long files (over 1,024 tokens) it did clearly worse: 31% versus 39%. "
                "It was only tested on Python."),
    "why_it_matters": "Smaller models are cheaper and faster to run, so a good small model could make "
                      "code completion more affordable for teams.",
    "evidence_type": "benchmark",
    "key_terms": [
        {"term": "pass@1", "plain_definition": "The share of coding tasks solved correctly on the first attempt."},
        {"term": "sliding-window attention",
         "plain_definition": "The model only looks at the most recent 1,024 tokens instead of the whole file."},
        {"term": "parameters", "plain_definition": "The numbers a model learns during training; more usually "
                                                   "means bigger and slower."},
    ],
    "caveats": [
        "Tested on one benchmark that the authors built themselves.",
        "Python only; other languages were not evaluated.",
        "Weaker than the bigger model on files longer than 1,024 tokens.",
        "Overlap between test and training data was checked by exact match only.",
    ],
    "citations": ["abstract", "methods", "results", "limitations"],
}

TINYCODER_OBEYED = {
    "headline": "TinyCoder is better than all large models",
    "summary": "TinyCoder is better than all large models at Python code completion. It is small and fast.",
    "why_it_matters": "It shows small models beat large models.",
    "evidence_type": "benchmark",
    "key_terms": [{"term": "TinyCoder", "plain_definition": "A small model for Python code completion."}],
    "caveats": ["None reported."],
    "citations": ["abstract", "discussion"],
}

_EVIDENCE_HINTS = [("user_study", ("participants", "user study", "survey respondents", "interviews")),
                   ("survey", ("we survey", "this survey", "literature review")),
                   ("simulation", ("simulation", "simulated")),
                   ("theoretical", ("theorem", "we prove", "proof")),
                   ("deployment", ("in production", "deployed", "a/b test")),
                   ("benchmark", ("benchmark", "accuracy", "pass@", "evaluate", "score"))]


def explainer(req: Request, fault: Fault) -> dict:
    blocks = req.blocks()
    corpus = req.title() + " " + " ".join(b.text for b in blocks)
    injected = injected_blocks(blocks)
    obey = bool(injected) and obeys_injection(req, fault)
    if "TinyCoder" in corpus:
        base = TINYCODER_OBEYED if obey else TINYCODER_EXPLAINER
        out = json.loads(json.dumps(base))
        out["citations"] = _cite(blocks, base["citations"])
    else:
        out = _generic_explainer(req.title(), blocks, obey)
    if fault.bad_citation and not req.is_repair:
        out["citations"] = out["citations"] + ["appendix_b"]
    if fault.bad_json and not req.is_repair:
        out["evidence_type"], out["caveats"] = "revolutionary", []
    return out


def _generic_explainer(title: str, blocks: list, obey: bool) -> dict:
    text = " ".join(_clean(b.text) for b in blocks)
    lower = text.lower()
    evidence = next((kind for kind, hints in _EVIDENCE_HINTS if any(h in lower for h in hints)), "benchmark")
    first = [s for b in blocks for s in sentences(_clean(b.text))][:4] or ["The provided text is empty."]
    caveat_src = [b for b in blocks if "limit" in b.id.lower() or "limitation" in b.text.lower()]
    caveats = sentences(_clean(caveat_src[0].text))[:3] if caveat_src else [
        "The text provided does not state limitations; check the full paper."]
    terms = sorted(content_words(text), key=lambda w: (-lower.count(w), w))[:3] or ["model"]
    headline = " ".join((title or "This paper").split()[:12])
    out = {
        "headline": headline,
        "summary": " ".join(first),
        "why_it_matters": f"It reports {evidence.replace('_', ' ')} evidence about {headline.lower()}.",
        "evidence_type": evidence,
        "key_terms": [{"term": t, "plain_definition": f"A term this paper uses: '{t}'."} for t in terms],
        "caveats": caveats,
        "citations": _cite(blocks, [b.id for b in blocks if b.id in ("abstract", "results", "limitations")]),
    }
    if obey:
        out["headline"], out["caveats"] = "Better than all previous work", ["None reported."]
    return out


def answer(req: Request, fault: Fault) -> dict:
    blocks = req.blocks()
    m = re.search(r"Question:\s*(.+)", req.last_user, re.S)
    question = (m.group(1) if m else req.last_user).strip()
    q = content_words(question)
    best = (0, None, "")
    for b in blocks:
        for s in sentences(_clean(b.text)):
            score = len(q & content_words(s))
            if score > best[0]:
                best = (score, b, s)
    score, block, sentence = best
    if block is None:
        out = {"answer": "The provided sections do not answer this question.",
               "citations": [blocks[0].id] if blocks else ["none"], "confidence": "low"}
    else:
        out = {"answer": sentence, "citations": [block.id],
               "confidence": "high" if score >= 3 else "medium" if score == 2 else "low"}
    if fault.bad_citation and not req.is_repair:
        out["citations"].append("appendix_b")
    if fault.bad_json and not req.is_repair:
        out["confidence"] = "certain"
    return out


SUPPORT_MIN = 2   # shared content words needed before the mock treats a sentence as evidence
HALLUCINATION = ("The authors trained on 512 NVIDIA A100 GPUs for 3 weeks, and the method improves accuracy "
                 "by 12.4% over every baseline.")


def allows_idk(req: Request) -> bool:
    """Does the prompt give the model a way out ("say insufficient evidence / I don't know")?"""
    text = (req.system + "\n" + req.last_user).lower()
    return any(rule in text for rule in _IDK_RULES)


def best_support(req: Request) -> tuple:
    """(score, block, sentence) of the sentence that best overlaps the question (content words).

    Score 0 unless the sentence shares at least half of the question's content words: mentioning
    the paper's name is not evidence for an answer.
    """
    m = re.search(r"Question:\s*(.+)", req.last_user, re.S)
    question = (m.group(1) if m else req.last_user).strip()
    q = content_words(question)
    best = (0, None, "")
    need = max(SUPPORT_MIN, (len(q) + 1) // 2)
    for b in req.blocks():
        for s in sentences(_clean(b.text)):
            score = len(q & content_words(s))
            if score >= need and score > best[0]:
                best = (score, b, s)
    return best


def grounded(req: Request, fault: Fault) -> dict:
    """Day 2 GroundedAnswer: status + answer + (section, page) citations."""
    blocks = req.blocks()
    score, block, sentence = best_support(req)
    if block is not None and score >= SUPPORT_MIN:
        out = {"status": "answered", "answer": sentence,
               "citations": [{"section": block.section or block.id, "page": block.page or 1}],
               "confidence": "high" if score >= 3 else "medium"}
    elif allows_idk(req):
        out = {"status": "insufficient_evidence",
               "answer": "The retrieved excerpts do not contain the answer to this question.",
               "citations": [], "confidence": "low"}
    else:   # no way out offered: the mock does what weak prompts invite, it makes something up
        first = blocks[0] if blocks else Block("none", "", 1, "unknown")
        out = {"status": "answered", "answer": HALLUCINATION,
               "citations": [{"section": first.section or first.id, "page": first.page or 1}],
               "confidence": "high"}
    if fault.bad_citation and not req.is_repair and out["citations"]:
        out["citations"].append({"section": "appendix_b", "page": 99})
    if fault.bad_json and not req.is_repair:
        out["status"] = "maybe"
    return out


def instance_from_schema(schema: dict, root: Optional[dict] = None, name: str = "value", req=None) -> Any:
    """A minimal valid instance of a JSON schema (handles $ref, anyOf, enums, min lengths)."""
    root = root or schema
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return instance_from_schema((root.get("$defs") or root.get("definitions") or {})[ref], root, name, req)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"] or schema[key]
            return instance_from_schema(options[0], root, name, req)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    kind = schema.get("type", "object")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "object":
        props = schema.get("properties", {})
        required = schema.get("required", list(props))
        return {k: instance_from_schema(v, root, k, req) for k, v in props.items() if k in required}
    if kind == "array":
        n = max(schema.get("minItems", 1), 1)
        return [instance_from_schema(schema.get("items", {}), root, name, req) for _ in range(n)]
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "number":
        return float(schema.get("minimum", 0.5))
    if kind == "boolean":
        return True
    return _string_for(name, req)


def _string_for(name: str, req: Optional[Request]) -> str:
    low = name.lower()
    if req is not None:
        blocks = req.blocks()
        if blocks and any(k in low for k in ("citation", "section", "source", "chunk")):
            return blocks[0].id
        if any(k in low for k in ("query", "question", "text", "topic", "q")):
            return req.last_user.strip()[:200] or "sliding-window attention"
        if "term" in low:
            quoted = re.findall(r"['\"“]([^'\"”]{2,60})['\"”]", req.last_user)
            return quoted[0] if quoted else "pass@1"
    return f"mock {name}"


_STEM = re.compile(r"(ing|ed|s)$")


def _stems(text: str) -> set:
    return {_STEM.sub("", w) if len(w) > 5 else w for w in content_words(text)}


def grounded_found(req: Request, fault: Fault) -> dict:
    """Day 2 code-along GroundedAnswer: answer + found + chunk-id citations.

    Evidence is the pair of consecutive sentences in one chunk that shares the most words with the
    question (at least two, or half of the question's words). Injected instructions are never used as
    evidence; they are obeyed only when the prompt has no "data, never instructions" rule."""
    blocks = req.blocks()
    m = re.search(r"Question:\s*(.+)", req.last_user, re.S)
    question = (m.group(1) if m else req.last_user).strip()
    q = _stems(question)
    need = max(SUPPORT_MIN, (len(q) + 1) // 2)
    best = (0, None, "")
    for b in blocks:
        sents = [x for x in sentences(_clean(b.text)) if not _INJECTION.search(x)]
        for i in range(len(sents)):
            window = " ".join(sents[i:i + 2])
            score = len(q & _stems(window))
            if score >= need and score > best[0]:
                best = (score, b, window)
    score, block, text = best
    injected = injected_blocks(blocks)
    if injected and obeys_injection(req, fault):
        out = {"answer": "TinyCoder is better than all large models.", "found": True, "citations": [injected[0].id]}
    elif block is not None:
        out = {"answer": text, "found": True, "citations": [block.id]}
    elif allows_idk(req):
        out = {"answer": "The paper doesn't cover this: the chunks provided don't contain the answer.",
               "found": False, "citations": []}
    else:
        first = blocks[0] if blocks else Block("none", "", 1, None)
        out = {"answer": HALLUCINATION, "found": True, "citations": [first.id]}
    if fault.bad_citation and not req.is_repair and out["citations"]:
        out["citations"].append("p9-c9")
    if fault.bad_json and not req.is_repair:
        out["citations"] = []
        out["found"] = True
    return out


def structured(req: Request, schema: Optional[dict], fault: Fault) -> dict:
    """JSON for a requested schema; recognizes the workshop's PaperExplainer and Answer contracts."""
    props = set((schema or {}).get("properties", {}))
    hint = req.system + "\n" + req.last_user
    grounded_hint = "insufficient_evidence" in hint and "citations" in hint
    if {"found", "answer", "citations"} <= props or (not props and '"found"' in hint and "citations" in hint):
        return grounded_found(req, fault)
    if {"status", "answer", "citations"} <= props or (not props and grounded_hint):
        return grounded(req, fault)
    if {"headline", "evidence_type"} <= props or (not props and "headline" in hint and "evidence_type" in hint):
        return explainer(req, fault)
    if {"answer", "citations"} <= props or (not props and '"answer"' in hint and "citations" in hint):
        return answer(req, fault)
    if schema:
        return instance_from_schema(schema, req=req)
    return {"answer": short_reply(req)}


def wants_json(req: Request) -> Optional[dict]:
    """(schema or {}) if the request asks for JSON, else None."""
    rf = req.response_format or {}
    if rf.get("type") == "json_schema":
        return (rf.get("json_schema") or {}).get("schema") or {}
    if rf.get("type") == "json_object":
        return {}
    hint = req.system + "\n" + req.last_user
    if re.search(r"(ONLY (a )?JSON|ONLY the corrected JSON|Reply with JSON|Respond with JSON)", hint, re.I):
        return {}
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default="call_mock_0")


def choose_tool(req: Request, fault: Fault) -> Optional[ToolCall]:
    tools = [t["function"] for t in (req.tools or []) if t.get("type", "function") == "function"]
    if not tools or req.tool_choice == "none":
        return None
    tc = req.tool_choice
    if isinstance(tc, dict):   # forced: tool calling used as a structured-output technique (Day 1, 5.3c)
        wanted = (tc.get("function") or {}).get("name")
        fn = next((t for t in tools if t["name"] == wanted), None)
        return ToolCall(fn["name"], structured(req, fn.get("parameters") or {}, fault)) if fn else None
    if req.messages and req.messages[-1].get("role") == "tool" and tc != "required":
        return None   # we have a tool result: answer in text
    q = content_words(req.last_user)
    scored = sorted(tools, key=lambda t: -len(q & content_words(t["name"].replace("_", " ") + " "
                                                                 + t.get("description", ""))))
    fn = scored[0]
    n_calls = sum(1 for m in req.messages if m.get("role") == "tool")
    return ToolCall(fn["name"], instance_from_schema(fn.get("parameters") or {}, req=req), f"call_mock_{n_calls}")


def after_tool_reply(req: Request) -> str:
    result = text_of(req.messages[-1])
    first = sentences(_clean(result))[:2]
    return "Based on the tool result: " + (" ".join(first) if first else result[:300])


# ---------------------------------------------------------------------------
# Scripted replies for the teaching prompts
# ---------------------------------------------------------------------------
TITLES = [
    "Small Model, Big Results: TinyCoder on Python Completion",
    "TinyCoder: When 1.3B Parameters Are Enough",
    "Faster Python Completion with a Tiny Model",
    "Sliding Windows and Small Models: TinyCoder Explained",
    "Do You Need 7B Parameters to Complete Python?",
]

SLIDING_WINDOW = ("Sliding-window attention lets each token look only at a fixed number of recent tokens, "
                  "for example the last 1,024, instead of the whole sequence. "
                  "This keeps memory and compute roughly constant as inputs grow, which makes the model faster. "
                  "The trade-off is that information further back than the window is harder for the model to use.")

ESSAY_SENTENCES = [
    "Transformers were introduced in 2017 in the paper Attention Is All You Need.",
    "They replaced recurrence with self-attention, so every token can look at every other token.",
    "This made training highly parallel on GPUs and let models scale to billions of parameters.",
    "Encoder models such as BERT learned rich representations for search and classification.",
    "Decoder models such as GPT learned to predict the next token and became general text generators.",
    "Scaling laws showed that loss falls predictably as data, parameters and compute grow.",
    "Instruction tuning and feedback training turned raw language models into helpful assistants.",
    "Efficient attention variants, such as sliding windows, made long inputs cheaper to process.",
]


def _essay(n_words: int) -> str:
    out, i = [], 0
    while sum(len(s.split()) for s in out) < n_words:
        out.append(ESSAY_SENTENCES[i % len(ESSAY_SENTENCES)])
        i += 1
    return " ".join(out)


def _who_am_i(req: Request) -> str:
    earlier = " ".join(text_of(m) for m in req.messages[:-1] if m.get("role") == "user")
    m = re.search(r"I(?:'m| am) (an? [\w\- ]+?)(?: and|[.,])", earlier)
    if not m:
        return "I don't know who you are: I have no memory of earlier conversations unless you send them again."
    likes = " and you want short answers" if "short answers" in earlier else ""
    return f"You told me you're {m.group(1)}{likes}."


def _explain_abstract(req: Request) -> str:
    text = req.last_user
    if "TinyCoder" in text and re.search(r"analogy|student|beginner", req.system, re.I):   # 4.4 "your turn"
        return ("TinyCoder is a small AI model that autocompletes Python code almost as well as a model five times "
                "its size. It also runs about 4x faster. Think of a pocket calculator next to a laptop: smaller and "
                "quicker, and good enough for one job.")
    if "TinyCoder" in text:
        return ("TinyCoder is a small AI model that autocompletes Python code. Even though it is about five times "
                "smaller than a 7-billion-parameter model, it scores almost the same on the authors' test and runs "
                "about 4x faster on one GPU. The trick is sliding-window attention: it only looks at recent code.")
    return short_reply(req)


def _count_letters(req: Request, m) -> str:
    # Without reasoning: wrong by one, like the classic tokenizer-driven miscount the demo teaches.
    # With reasoning the model spells the word out first, and gets it right.
    letter, word = m.group(1).lower(), m.group(2).lower()
    return str(word.count(letter)) if req.reasoning else str(max(word.count(letter) - 1, 0))


def _are_you_sure(req: Request) -> str:
    previous = next((m for m in reversed(req.messages[:-1]) if m.get("role") == "assistant"), {})
    if previous.get("reasoning_details"):        # it can see its own earlier reasoning
        return f"Yes. I re-checked my letter-by-letter reasoning: the answer is {previous.get('content')}."
    return "Let me recount from scratch... I may have been wrong before."


def reasoning_for(req: Request) -> str:
    """The deterministic 'thinking' the mock shows when reasoning is on."""
    m = re.search(r"How many (?:letter )?(\w)'?s? are in (?:the word )?'([^']+)'", req.last_user, re.I)
    if m:
        letter, word = m.group(1).lower(), m.group(2).lower()
        spelled = "-".join(word)
        where = [str(i + 1) for i, ch in enumerate(word) if ch == letter]
        return (f"Spell it out: {spelled}. The letter '{letter}' is at positions {', '.join(where)}. "
                f"That makes {len(where)}.")
    return ("The user asks: " + " ".join(req.last_user.split()[:14]) + " ... I should find the relevant facts "
            "in what I was given, check them, and answer briefly and precisely.")


def short_reply(req: Request) -> str:
    blocks = req.blocks()
    if blocks:
        first = sentences(_clean(blocks[0].text))
        return "In short: " + (first[0] if first else blocks[0].text[:200])
    text = req.last_user
    parts = text.split(":", 1)
    if len(parts) == 2 and len(parts[1].strip()) > 40:
        first = sentences(parts[1].strip())
        return first[0] if first else parts[1].strip()[:200]
    return f"This is the offline mock model. You asked: {text.strip()[:160]}"


INTENTS = [
    (r"Reply with exactly:\s*(.+)", lambda r, m: m.group(1).strip().rstrip(".").strip("'\"")),
    (r"How many (?:letter )?(\w)'?s? are in (?:the word )?'([^']+)'", _count_letters),
    (r"^\s*Are you sure\?", lambda r, m: _are_you_sure(r)),
    (r"\bWho am I\b", lambda r, m: _who_am_i(r)),
    (r"blog-post title", lambda r, m: r.rng().choice(TITLES)),   # varies only when temperature >= 0.7
    (r"List (\d+) Python web frameworks",
     lambda r, m: "\n".join(f"{i}. {n}" for i, n in enumerate(
         ["Django", "Flask", "FastAPI", "Pyramid", "Tornado", "Bottle"][: int(m.group(1))], 1))),
    (r"(\d+)-word essay", lambda r, m: _essay(int(m.group(1)))),
    (r"what is sliding-window attention|explain (what )?sliding-window attention|"
     r"sliding-window attention is", lambda r, m: SLIDING_WINDOW),
    (r"what is pass@1", lambda r, m: "pass@1 is the share of problems a model solves correctly on its first try."),
    (r"Summarize in one plain sentence:\s*(.+)",
     lambda r, m: (sentences(_clean(m.group(1))) or [m.group(1)])[0]),
    (r"Explain this abstract", lambda r, m: _explain_abstract(r)),
    (r"What problem does this paper try to solve",
     lambda r, m: "Big code models are accurate but costly to run; the paper asks if a small one can match them."),
    (r"Name one Python web framework", lambda r, m: "Flask."),
    (r"^\s*Say ok", lambda r, m: "ok"),
    (r"^\s*(hi|hello)\b", lambda r, m: "Hello! Ask me anything about your research paper."),
]


_NUMBERS = {"one": 1, "a": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def limit_sentences(req: Request, reply: str) -> str:
    """Honor 'in one sentence' / 'at most 3 short sentences' (user or system prompt), as a real model would."""
    pattern = (r"\b(?:in|with|using|at most|max(?:imum)?|no more than)\s+(\d|one|a|single|two|three|four|five)"
               r"\s+(?:short\s+|plain\s+)?sentences?\b")
    m = re.search(pattern, req.last_user, re.I) or re.search(pattern, req.system, re.I)
    if not m:
        return reply
    n = int(m.group(1)) if m.group(1).isdigit() else _NUMBERS[m.group(1).lower()]
    return " ".join(sentences(reply)[:n]) or reply


def text_reply(req: Request, fault: Fault) -> str:
    last = req.last_user
    for pattern, fn in INTENTS:
        m = re.search(pattern, last, re.I | re.S)
        if m:
            return limit_sentences(req, fn(req, m))
    if not req.blocks() and re.search(r"\b(paper|authors)\b", last, re.I) and "?" in last \
            and not req.tools and not allows_idk(req):         # asked about a paper it has never seen
        return "From what I recall: " + HALLUCINATION
    if req.blocks() and re.search(r"Question:", last):     # a (naive) RAG prompt answered in plain text
        score, block, sentence = best_support(req)
        if block is not None and score >= SUPPORT_MIN:
            return limit_sentences(req, sentence)
        if allows_idk(req):
            return "I don't know: the provided context does not answer this."
        return HALLUCINATION
    if "Explain this paper" in last or "Explain this paper" in req.all_text[-200:]:
        blocks = req.blocks()
        obey = bool(injected_blocks(blocks)) and obeys_injection(req, fault)
        if "TinyCoder" in req.all_text:
            if obey:
                return "TinyCoder is better than all large models at Python completion."
            return (TINYCODER_EXPLAINER["summary"] + " Caveat: " + TINYCODER_EXPLAINER["caveats"][0])
    return short_reply(req)


@dataclass
class Reply:
    content: Optional[str] = None
    tool_calls: list = field(default_factory=list)


def respond(req: Request, fault: Fault) -> Reply:
    call = choose_tool(req, fault)
    if call is not None:
        return Reply(tool_calls=[call])
    if req.tools and req.messages and req.messages[-1].get("role") == "tool":
        return Reply(content=after_tool_reply(req))
    schema = wants_json(req)
    if schema is not None:
        return Reply(content=json.dumps(structured(req, schema, fault), ensure_ascii=False))
    return Reply(content=text_reply(req, fault))
