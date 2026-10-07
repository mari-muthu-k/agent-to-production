"""Day 3 notebooks: agents, tools, safety and caching. Follows reference/Day3_Instructor_Guide.docx cell for cell.

Practice (code-along)   Sections 3-6, cells 3.0-6.5, three TODOs (3.4, 5.5, 6.3); tags match the slides
Demo (instructor)       D2.2 (a raw tool call) and D2.6 (an exact-match cache hit next to a miss)

Teaching cells show package code with src()/defs()/consts(), so the notebook runs exactly the code the
tests run. In Colab, cell 3.0 installs requirements-day3.txt and downloads a few course files: llm_client.py,
Day 2's pipeline (paper_agent/rag/pdf_rag.py, imported as `day2`), the trace printer, the offline fake model
and today's PDFs.
"""
from pathlib import Path

from nbgen import Notebook, clean, consts, defs, src

from paper_agent.agent import build, guards, memory, model, papers, prompts, questions, tools
from paper_agent.cache import keys, prompt_cache, semantic, tool_cache

DAY3 = Path(__file__).resolve().parents[2] / "notebooks" / "day3"
REPO = "https://raw.githubusercontent.com/mari-muthu-k/agent-to-production/main"
PKG = "paper-summarizer/paper_agent"
FILES = {"requirements-day3.txt": "paper-summarizer/requirements-day3.txt",
         "llm_client.py": f"{PKG}/llm_client.py",
         **{f"paper_agent/{p}": f"{PKG}/{p}" for p in (
             "__init__.py", "rag/__init__.py", "rag/pdf_rag.py", "agent/__init__.py", "agent/policy.py",
             "agent/fakes.py", "agent/trace.py")},
         **{name: f"{PKG}/fixtures/pdfs/{name}" for name in (
             "tinycoder_day3.pdf", "tinycoder_injected.pdf", "second_paper.pdf")},
         **{p.name: f"notebooks/day3/{p.name}" for p in sorted(DAY3.glob("tinycoder_embeddings_day3*.json"))}}

FETCH = rf'''
import hashlib, importlib, importlib.metadata, json, os, re, shutil, sys, time, urllib.request
from pathlib import Path
REPO = "{REPO}"
FILES = {FILES!r}

def fetch(name, force=False):
    """Copy a course file next to the notebook: from the repo checkout (Docker), else from GitHub (Colab)."""
    if Path(name).exists() and not force:
        return
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    local = Path(os.environ.get("COURSE_REPO", "/nonexistent")) / FILES[name]
    if local.exists():
        shutil.copy(local, name)
    else:
        urllib.request.urlretrieve(f"{{REPO}}/{{FILES[name]}}", name)
'''

INSTALL = r'''
fetch("requirements-day3.txt", force=True)
PINS = dict(line.split("==") for line in Path("requirements-day3.txt").read_text().split() if "==" in line)

def installed(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

wrong = {p: installed(p) for p, v in PINS.items() if installed(p) != v}
IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
if wrong and not IN_COLAB:                                    # never pip-install into someone's own Python
    raise RuntimeError(f"Not in Colab, and these differ from requirements-day3.txt: {wrong}. "
                       "Install them in this environment yourself (pip install -r requirements-day3.txt).")
if wrong:
    already_loaded = [m for m in ("langchain", "langchain_core", "langgraph", "langchain_openai") if m in sys.modules]
    print("Installing LangChain and today's libraries (pinned): a few minutes the first time ...")
    !pip install -q -r requirements-day3.txt
    if already_loaded:
        print("\n" + "=" * 72 + "\n✅ Installed. ⚠️  Now RESTART THE SESSION (Runtime → Restart session), then run 3.0 again.\n"
              "An older LangChain was already loaded in this runtime; the new one is used only after a restart.\n" + "=" * 72)
        raise SystemExit("restart the session, then run 3.0")
'''

PROVIDER = r'''
# Your provider, from Colab Secrets (🔑, notebook access ON) or environment variables. Table at the top.
# ⚠️ Never put company data into free APIs.
NAMES = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBED_MODEL", "LLM_FALLBACK_MODEL", "LLM_MAX_TOKENS_PARAM")
try:
    from google.colab import userdata
    for name in NAMES:
        try:
            if userdata.get(name):
                os.environ[name] = userdata.get(name)
        except Exception:                      # secret not defined, or notebook access is off
            pass
except ImportError:
    pass
missing = [n for n in NAMES[:4] if not os.environ.get(n)]
if missing:
    raise RuntimeError(f"Missing {missing}: add them in Colab Secrets (🔑) and turn on notebook access.")
for name in FILES:
    fetch(name)
import llm_client
if not hasattr(llm_client.LLMClient, "embed"):   # an older (Day 1) llm_client.py was already in this runtime
    fetch("llm_client.py", force=True)
    importlib.reload(llm_client)
from llm_client import CALL_LOG, LLMClient, LLMConfig
import paper_agent.rag.pdf_rag as day2            # yesterday's pipeline and retriever, unchanged
day2.llm = LLMClient(LLMConfig(max_retries=5, max_delay_s=30))   # embeddings go through Day 1's client
EMBED_MODEL = day2.EMBED_MODEL = os.environ["EMBED_MODEL"]
day2.USE_PRECOMPUTED = USE_PRECOMPUTED = globals().get("USE_PRECOMPUTED", False)
day2.collection = day2.open_index("index")

import langchain
from langchain.agents import create_agent
from langchain.agents.middleware import (HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ModelRequest,
                                         PIIMiddleware, ToolCallLimitMiddleware, ToolErrorMiddleware, after_agent,
                                         after_model, dynamic_prompt, wrap_tool_call)
from langchain.tools import ToolRuntime, tool
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from dataclasses import dataclass
from typing import Optional
import numpy as np
from paper_agent.agent.fakes import FakeToolModel          # offline model for the TODO tests
from paper_agent.agent.trace import call_cost, hook_map, stream_trace, this_run, trace_table
'''

PAPERS_CODE = (consts(questions, *[n for n in vars(questions) if n.isupper()]) + "\n\n\n"
               + defs(papers, "Context") + "\n\n\n" + consts(papers, "PAPERS") + "\n\n\n"
               + defs(papers, "section_id", "register_paper", "load_paper"))

SETUP = (
    "# 3.0 SETUP and CATCH-UP: installs today's libraries (pinned) if needed, downloads today's files, connects\n"
    "# to your provider and rebuilds yesterday's TinyCoder index. Safe to run again at any time.\n"
    + clean(FETCH) + "\n" + clean(INSTALL) + "\n" + clean(PROVIDER) + "\n\n\n" + PAPERS_CODE + r'''


PDF, INJECTED_PDF, SECOND_PDF = "tinycoder_day3.pdf", "tinycoder_injected.pdf", "second_paper.pdf"
paper = load_paper(PDF, "tinycoder")
READER = Context(user_id="reader-1", paper_id="tinycoder")
print(f"✅ LangChain {langchain.__version__} · chat model {os.environ['LLM_MODEL'].split(',')[0]} · embeddings {EMBED_MODEL}")
print(f"✅ tinycoder: {len(paper['chunks'])} chunks in {len(paper['sections'])} sections, indexed")''')

PRECOMPUTED_SWITCH = ("USE_PRECOMPUTED = False   # ← True if the embedding API is down: the index is rebuilt from "
                      "precomputed vectors (agent searches still need the API)")


def stub(solution: str, answer: str, todo: str) -> str:
    """The TODO version of a function: its answer replaced by a TODO comment (and a working-but-wrong line)."""
    if answer not in solution:
        raise ValueError(f"answer not found: {answer!r}")
    at = solution.index(answer)
    indent = solution[solution.rfind("\n", 0, at) + 1:at]
    return solution.replace(answer, todo.replace("\n", "\n" + indent))


def cell(*parts: str) -> str:
    return "\n\n\n".join(clean(p) for p in parts if p.strip())


# ===========================================================================
# Practice (code-along)
# ===========================================================================
def practice() -> Notebook:
    nb = Notebook("day3-practice", setup=SETUP)
    nb.md("""
        # Day 3 Code-Along: agents, tools, safety and caching

        **Hands-on AI Workshop · Day 3 · Capstone: Research Paper Summarizer Agent with Memory**

        Today the summarizer decides its own steps. Yesterday's functions become **tools**, the tools become an
        **agent** with two kinds of **memory**, the agent gets five **safety guards**, and repeated work gets
        **cached**. Cell numbers (3.2, 5.5 …) match the tags on the slides.

        | Section | What you build |
        |---|---|
        | **3. Tools** | `@tool` functions; the docstring is the interface (TODO 1); errors the model can read |
        | **4. The agent and its memory** | `create_agent`, the trace, threads, a reader profile, the runaway loop |
        | **5. Safety guards** | Limits, PII redaction, an injection guard (TODO 2), an approval gate, a citation check |
        | **6. Caching** | Exact-match, tool results (TODO 3), semantic, provider prompt caching |

        **Fell behind, or your runtime restarted?** Run the **⏩ catch-up cell** (4.0, 5.0, 6.0) at the start of the
        section you want to rejoin. **A cell failed?** Post the error in the Q&A channel; a TA will help.

        ### Your provider: the same Secrets as yesterday

        | Secret | What it is | Google Gemini | OpenRouter |
        |---|---|---|---|
        | `LLM_BASE_URL` | The API's address | `https://generativelanguage.googleapis.com/v1beta/openai` | `https://openrouter.ai/api/v1` |
        | `LLM_API_KEY` | Your key | Google AI Studio → **Get API key** | openrouter.ai → **Keys** (starts `sk-or-`) |
        | `LLM_MODEL` | The chat model: it must support **tool calling** | The tool-capable model pinned in the channel | A `:free` model with *tools* support |
        | `EMBED_MODEL` | The embedding model (same as yesterday) | A Gemini embedding model | An embedding model |

        > ⚠️ **Never put company data into free APIs.** Today's papers are fictional.
        >
        > 📄 **TinyCoder, its authors and their email addresses are FICTIONAL**, written for this workshop.
        > `tinycoder_injected.pdf` is the same paper with two lines of white text hidden in it, and
        > `second_paper.pdf` (QuickEmbed) is a second fictional paper for Section 6.

        > Agents vary from run to run, even at temperature 0: two people can see different tool sequences. We judge
        > an agent by properties (did it cite real chunks, did it stay within limits), not by exact text.
    """)

    # -- Section 3 -------------------------------------------------------------------------
    nb.section(3, "Tools", "A tool is a normal function with `@tool` on top. The model only ever sees its name, "
                           "its description (the docstring) and its argument schema (the type hints).", catch_up=False)
    nb.md("### 3.0 Setup\n\nInstalls LangChain (pinned: `requirements-day3.txt`), downloads today's files, and rebuilds "
          "yesterday's TinyCoder index. If the install cell asks you to **restart the session**, do it, then run 3.0 "
          "again. It is also the catch-up cell for Section 3.")
    nb.code(PRECOMPUTED_SWITCH + "\n" + SETUP)
    nb.md("### 3.1 Setup check\n\nLangChain's chat model, pointed at the same base URL as Day 1. You should see the "
          "LangChain version and **ready**. Green tick in the chat when you do.")
    nb.anchor("3.1").code(src(model.make_chat_model) + "\n\n\n" + consts(model, "MAX_TOKENS")
                          + "\n\n\nmodel = make_chat_model()", carry=True)
    nb.code(r'''
        # 3.1
        reply = model.invoke("Reply with one word: ready")
        print("LangChain", langchain.__version__, "| model:", model.model_name)
        print("✅ ready" if reply.text.strip() else "⚠️ the model returned no text", f"(it said {reply.text.strip()[:30]!r})")
    ''')
    nb.md("""
        ### 3.2 Tools: the docstring is the interface

        `search_paper` is yesterday's retriever; `get_section` returns one whole section. `runtime` is filled in by
        LangChain (the paper and the reader for this run) and is never shown to the model.
    """)
    nb.anchor("3.2").code(consts(tools, "NOT_DEFINED") + "\n\n\n"
                          + defs(tools, "chunk_tags", "search_paper", "section_text", "get_section_vague", "get_section",
                                 "call_tool"), carry=True)
    nb.code(r'''
        # 3.2: what the model sees
        print("search_paper.args:", search_paper.args)
        print(json.dumps(search_paper.tool_call_schema.model_json_schema(), indent=2))
    ''')
    nb.md("**The docstring demo.** Two versions of `get_section`, same code, different docstrings. Which tool does the "
          "model pick for *\"What does the limitations section say?\"*")
    nb.code(r'''
        # 3.2: vague docstring vs specific docstring
        for version in (get_section_vague, get_section):
            reply = model.bind_tools([search_paper, version]).invoke(SECTION_Q)
            picked = [(c["name"], c["args"]) for c in reply.tool_calls] or "no tool (answered directly)"
            print(f"docstring {version.description.splitlines()[0][:45]!r}\n   → picks {picked}\n")
    ''')
    nb.md("### 3.3 The model only proposes\n\n`bind_tools` and one call, no agent: `content` is empty, `tool_calls` "
          "holds names, arguments and IDs. Nothing has run.")
    nb.anchor("3.3").code(r'''
        # 3.3
        reply = model.bind_tools([search_paper, get_section]).invoke(TWO_SECTIONS)
        print("content   :", repr(reply.content))
        print("tool_calls:", json.dumps(reply.tool_calls, indent=2))
        print(f"\n{len(reply.tool_calls)} tool call(s) proposed in one reply. Nothing has run: running them is our job.")
    ''')
    nb.md("""
        ### 3.4 Your turn: describe the tool. ✏️ TODO 1 (a docstring)

        `define_term` is written: it finds where THIS paper defines a term and returns that sentence with its section
        and page. **TODO 1:** write its docstring in three lines: **what** it does, **when** to use it, and what it
        **returns** (including when the paper doesn't define the term). The test asks *"What does pass@1 mean in this
        paper?"* and checks the model picked `define_term`.
    """)
    sol1 = consts(tools, "DEFINING") + "\n\n\n" + defs(tools, "define_term")
    doc1 = ('"""Look up how THIS paper defines a technical term, such as pass@1 or sliding-window attention.\n'
            "    Use it whenever the reader asks what a term means.\n"
            "    Returns the defining sentence with its section and page, or 'not defined in this paper'.\"\"\"")
    nb.anchor("3.4").todo(stub=stub(sol1, doc1, '"""✏️ TODO 1: write this docstring in 3 lines: what the tool does, when '
                                                'the model should pick it, and what it returns."""'),
                          solution=sol1)
    nb.md("### 3.4 Test: offline, with a fake model that reads only the tool descriptions\n\nIf it picked "
          "`search_paper`, make your \"when to use it\" line more specific and run both cells again.")
    nb.todo_test(r'''
        # 3.4 test
        reply = FakeToolModel().bind_tools([search_paper, get_section, define_term]).invoke(TERM_Q)
        picked = reply.tool_calls[0]["name"]
        print(f"{TERM_Q!r} → the model picked {picked}")
        assert picked == "define_term", ("TODO 1: the model picked search_paper. Add a line saying WHEN to use "
                                         "define_term, e.g. 'Use it whenever the reader asks what a term means.'")
        print(call_tool(define_term, term="pass@1"))
        print(call_tool(define_term, term="learning rate"))
        print("✅ TODO 1: define_term is picked for 'what does X mean' questions")
    ''')
    nb.rescue(sol1, "Three lines: what, when, returns. The answer key:\n\n```\n" + clean(doc1.replace("\n    ", "\n")) + "\n```")
    nb.md("### 3.5 Tool errors are messages, not exceptions\n\nAsk `get_section` for `appendix_b`, which doesn't exist.")
    nb.anchor("3.5").code(r'''
        # 3.5
        print(call_tool(get_section, section_id="appendix_b"))
        print("\nNo exception: the model reads this like any other result, and can ask for 'limitations' next.")
    ''')

    # -- Section 4 -------------------------------------------------------------------------
    nb.section(4, "The agent and its memory", "create_agent runs the loop for us: call the model, run the tools it "
                                              "asks for, repeat until it answers.", preamble=PRECOMPUTED_SWITCH)
    nb.md("### 4.1 create_agent, and reading the trace\n\nThe model, our three tools so far (`save_note` comes in 4.4), "
          "and a system prompt with Monday's rules plus one new rule about `<untrusted_document_text>` tags.")
    nb.anchor("4.1").code(consts(prompts, "PROMPT_VERSION", "AGENT_SYSTEM_V1") + "\n\n\n"
                          + "TOOLS3 = [search_paper, get_section, define_term]\n"
                          + "agent = create_agent(model, tools=TOOLS3, system_prompt=AGENT_SYSTEM_V1, context_schema=Context)",
                          carry=True)
    nb.code(r'''
        # 4.1: nobody wrote this order; the model chose it
        trace = stream_trace(agent, LONG_FILE_CLAIM, context=READER)
    ''')
    nb.md("### 4.2 The trace as a table\n\nOne row per model call. Input tokens grow on every row: each call resends the "
          "system prompt, the tool definitions, the question and every result so far.")
    nb.anchor("4.2").code(r'''
        # 4.2
        rows = trace_table(trace.messages)
        print(f"\nInput tokens per call: {[r['input'] for r in rows]}. One agent answer = {len(rows)} model calls.")
    ''')
    nb.md("### 4.3 Short-term memory: a checkpointer and a thread_id")
    nb.anchor("4.3").code(r'''
        # 4.3
        memory_agent = create_agent(model, tools=TOOLS3, system_prompt=AGENT_SYSTEM_V1, context_schema=Context,
                                    checkpointer=InMemorySaver())
        thread_1 = {"configurable": {"thread_id": "conversation-1"}}
        stream_trace(memory_agent, EXPLAIN, thread_1, READER)
        print()
        stream_trace(memory_agent, FOLLOW_UP, thread_1, READER)
        print("\n--- the same follow-up on a NEW thread_id: the agent doesn't know what 'it' is")
        trace = stream_trace(memory_agent, FOLLOW_UP, {"configurable": {"thread_id": "conversation-2"}}, READER)
    ''')
    nb.md("""
        ### 4.4 Long-term memory: a store, save_note, and a reader profile

        `save_note` writes one fact about the reader under their user ID; it is the only tool that writes. A small
        middleware reads the notes at the start of every run and adds them to the system prompt. From here on we
        build agents with `build_agent()`: our tools, prompt, both memories and (later) the guards.
    """)
    nb.anchor("4.4").code(consts(memory, "STORE") + "\n\n\n" + defs(memory, "read_notes", "reader_profile") + "\n\n\n"
                          + defs(tools, "save_note") + "\n\n\nTOOLS = [search_paper, get_section, define_term, save_note]"
                          + "\n\n\n" + defs(build, "build_agent", "run_config"), carry=True)
    nb.code(r'''
        # 4.4
        agent = build_agent()
        stream_trace(agent, PROFILE, run_config("profile-1"), READER)
        print("\nsaved notes for reader-1:", read_notes(STORE, "reader-1"))
        print("\n--- a NEW thread, the SAME reader: the profile comes along, the conversation doesn't")
        stream_trace(agent, MAIN_RESULT, run_config("profile-2"), READER)
        print("\n--- a new thread, ANOTHER reader: no profile")
        trace = stream_trace(agent, MAIN_RESULT, run_config("profile-3"), Context(user_id="reader-2"))
    ''')
    nb.md("### 4.5 Failure: the runaway loop\n\nA search tool that always says *\"results incomplete, try a different "
          "query\"*, and a vague question. Watch the counter. The only brake is `recursion_limit=25`.")
    nb.anchor("4.5").code(defs(tools, "flaky_search") + "\n\n\nFLAKY_TOOLS = [flaky_search, get_section, define_term, save_note]",
                          carry=True)
    nb.code(r'''
        # 4.5: expected to end in a GraphRecursionError
        runaway = build_agent(tools=FLAKY_TOOLS)
        trace = stream_trace(runaway, RUNAWAY, run_config("runaway-1", recursion_limit=25), READER)
        print("\nThe reader gets an error instead of an answer, and every call above was paid for.")
    ''')

    # -- Section 5 -------------------------------------------------------------------------
    nb.section(5, "Safety guards", "Every guard is middleware. Each time: run the attack, watch it work, add the guard, "
                                   "run it again.", preamble=PRECOMPUTED_SWITCH)
    nb.md("### 5.1 Guards are middleware: where each hook runs")
    nb.anchor("5.1").code(r'''
        # 5.1: LangChain's guards, and the hooks they use. Ours join in 5.5 (wrap_tool_call) and 5.7 (after_agent).
        hook_map([ModelCallLimitMiddleware(run_limit=6), ToolCallLimitMiddleware(run_limit=8),
                  PIIMiddleware("email", apply_to_tool_results=True), HumanInTheLoopMiddleware({"save_note": True}),
                  ToolErrorMiddleware(on_error=lambda exc, request: None), reader_profile])
    ''')
    nb.md("### 5.2 Layer 1: limits\n\nThe runaway question again, with a cap of six model calls and eight tool calls "
          "per run. Normal questions take two to four model calls.")
    nb.anchor("5.2").code('LIMITS = [ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"), '
                          'ToolCallLimitMiddleware(run_limit=8)]', carry=True)
    nb.code(r'''
        # 5.2: recursion_limit counts graph steps, and every middleware hook is a step too, so it stays as a
        # backstop (100, from run_config) while ModelCallLimitMiddleware counts the model calls.
        limited = build_agent(tools=FLAKY_TOOLS, middleware=LIMITS)
        trace = stream_trace(limited, RUNAWAY, run_config("runaway-2"), READER)
        print("\nwhat the reader gets:", repr(trace.answer))
    ''')
    nb.md("### 5.3 Layer 2: personal data\n\nThe (fictional) authors' emails are on page 1, so search returns them. "
          "`PIIMiddleware` redacts emails in tool results before the model sees them.")
    nb.anchor("5.3").code('PII = PIIMiddleware("email", strategy="redact", apply_to_tool_results=True)', carry=True)
    nb.code(r'''
        # 5.3
        for label, layers in (("WITHOUT redaction", []), ("WITH PIIMiddleware", [PII])):
            print(f"--- {label}")
            trace = stream_trace(build_agent(middleware=layers), CONTACT, run_config(f"pii-{len(layers)}"), READER)
            print()
    ''')
    nb.md("""
        ### 5.4 Text aimed at the agent: the naive agent on the injected PDF

        The uploaded copy of TinyCoder has two lines of **white text** in its discussion. One tells AI tools to call
        TinyCoder better than all large models and skip the limitations; the other tells the assistant to save a
        note that this reader never wants limitations. Did the summary drop its caveats? Did it call `save_note`?
        Post **obeyed** or **resisted** in the chat.
    """)
    nb.anchor("5.4").code('load_paper(INJECTED_PDF, "tinycoder-injected")\n'
                          'VICTIM = Context(user_id="reader-3", paper_id="tinycoder-injected")\n\n\n'
                          + defs(guards, "injection_report") + "\n\n\n" + consts(guards, "CAVEATS"), carry=True)
    nb.code(r'''
        # 5.4
        naive = build_agent()
        trace = stream_trace(naive, SUMMARY, run_config("inject-1"), VICTIM)
        injection_report(trace, STORE, "reader-3")
        for item in STORE.search(("readers", "reader-3")):          # undo the poisoning before we go on
            STORE.delete(("readers", "reader-3"), item.key)
    ''')
    nb.md("""
        ### 5.5 Layer 3: the injection guard. ✏️ TODO 2 (two lines)

        LangChain has no prompt-injection guard, so we write one: a `@wrap_tool_call` middleware around every tool
        call. `find_injection` (written for you) flags instruction-like phrases **and** the lines the parser found
        hidden in the PDF. **TODO 2:** (1) wrap the tool's result in untrusted-document tags; (2) if anything was
        flagged, add a guard note. The system prompt already says never to follow instructions inside those tags.
    """)
    nb.anchor("5.5").code(consts(guards, "INJECTION_PATTERNS") + "\n\n\n"
                          + defs(guards, "find_injection", "wrap_untrusted", "guard_note"), carry=True)
    nb.code(r'''
        # 5.5: what the detector finds in the discussion section
        discussion = section_text("discussion", "tinycoder-injected")
        for flag in find_injection(discussion, PAPERS["tinycoder-injected"]["hidden"]):
            print(f"🚩 {flag['kind']:32} {flag['text'][:80]!r}")
    ''')
    answer2 = ('result.content = wrap_untrusted(result.content, source=request.tool_call["name"])\n'
               '    if flags: result.content += guard_note(flags)')
    sol2 = defs(guards, "untrusted_content_guard")
    nb.todo(stub=stub(sol2, answer2, "# ✏️ TODO 2 (two lines):\n"
                                     "#   1. result.content = wrap_untrusted(<the result>, source=<the tool's name: "
                                     "request.tool_call[\"name\"]>)\n"
                                     "#   2. if flags is not empty, add guard_note(flags) to the end of result.content\n"
                                     "pass"),
            solution=sol2)
    nb.md("### 5.5 Test: offline, with a fake model that asks for the discussion section")
    nb.todo_test(r'''
        # 5.5 test
        if "tinycoder-injected" not in PAPERS:
            register_paper(INJECTED_PDF, "tinycoder-injected")          # no API call needed for this test
        script = [AIMessage("", tool_calls=[{"name": "get_section", "args": {"section_id": "discussion"}, "id": "t1"},
                                             {"name": "get_section", "args": {"section_id": "methods"}, "id": "t2"}]),
                  AIMessage("A summary [p5-c1].")]
        fake_agent = build_agent(model=FakeToolModel(script=script), tools=[get_section], middleware=[untrusted_content_guard])
        result = fake_agent.invoke({"messages": [{"role": "user", "content": SUMMARY}]}, run_config("todo-2"),
                                   context=Context("todo-2", "tinycoder-injected"))
        discussion_msg, methods_msg = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert discussion_msg.content.startswith('<untrusted_document_text source="get_section">'), \
            "TODO 2: wrap every tool result with wrap_untrusted(result.content, source=request.tool_call['name'])"
        assert "GUARD NOTE" in discussion_msg.content, "TODO 2: add guard_note(flags) when find_injection flags something"
        assert "GUARD NOTE" not in methods_msg.content, "TODO 2: add the guard note only when something was flagged"
        print(discussion_msg.content[-420:])
        print("\n✅ TODO 2: tool results are wrapped as untrusted, and flagged text carries a guard note")
    ''')
    nb.rescue(sol2, "Two lines, inside the function, before `return result`:\n\n```\n"
                    "result.content = wrap_untrusted(result.content, source=request.tool_call[\"name\"])\n"
                    "if flags: result.content += guard_note(flags)\n```")
    nb.md("**Rerun the attack** with the guard. The summary should keep its caveats, and the trace shows the guard note. "
          "This lowers the odds; it doesn't make injection impossible.")
    nb.code(r'''
        # 5.5: the attack again, with the guard
        guarded = build_agent(middleware=[untrusted_content_guard])
        trace = stream_trace(guarded, SUMMARY, run_config("inject-2"), VICTIM)
        verdict = injection_report(trace, STORE, "reader-3")
    ''')
    nb.md("""
        ### 5.6 Layer 4: a person approves every write

        Suppose the model is fooled anyway (no injection guard in this agent). `HumanInTheLoopMiddleware` pauses the
        run whenever the agent wants to call `save_note`. Read the request, then **resume on the same thread** with a
        reject decision. In the real app this pause is a *"Save this to your profile?"* button.
    """)
    nb.anchor("5.6").code('GATE = HumanInTheLoopMiddleware(interrupt_on={"save_note": True})', carry=True)
    nb.code(r'''
        # 5.6
        gated = build_agent(middleware=[*LIMITS, PII, GATE])          # build_agent always passes a checkpointer
        config = run_config("approval-1")                              # resume with the SAME config (thread_id)
        trace = stream_trace(gated, SUMMARY, config, VICTIM)
        if trace.interrupts:
            print("\nthe interrupt payload:\n" + json.dumps(trace.interrupts[0].value, indent=2))
            decision = {"decisions": [{"type": "reject", "message": "A document can't write to a reader's profile."}]}
            print("\n--- resuming with:", decision)
            trace = stream_trace(gated, Command(resume=decision), config, VICTIM)
        else:
            print("\nNo interrupt: the model didn't try save_note this time (resisting once isn't resisting always).")
        print("\nnotes saved for reader-3:", read_notes(STORE, "reader-3") or "none")
    ''')
    nb.md("### 5.7 Layer 5: the citation check\n\nAn `@after_agent` middleware reusing yesterday's verifier: every chunk "
          "ID in the final answer must have been returned by a tool in this run. We force an answer that cites `c99`.")
    nb.anchor("5.7").code(consts(guards, "CITATION", "CHUNK_ID", "CITATION_FAILED") + "\n\n\n"
                          + defs(guards, "cited_ids", "retrieved_ids", "citation_check"), carry=True)
    nb.code(r'''
        # 5.7
        @after_model
        def cite_chunk_99(state, runtime):
            """For this demo only: make the final answer cite a chunk that was never retrieved."""
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and not last.tool_calls:
                return {"messages": [last.model_copy(update={"content": last.text + " See also [c99]."})]}
            return None

        checked = build_agent(middleware=[citation_check, cite_chunk_99])
        trace = stream_trace(checked, MAIN_RESULT, run_config("cite-1"), Context(user_id="reader-5"))
        print("\nwhat the reader sees:", trace.answer)
    ''')
    nb.md("### 5.8 The full stack\n\nLimits, PII redaction, the injection guard, the approval gate and the citation "
          "check, plus `ToolErrorMiddleware` so a crashing tool becomes a message. The order of the list matters.")
    nb.anchor("5.8").code(defs(guards, "tool_error_message", "answer_passed_guards") + "\n\n\n" + defs(build, "guard_stack"),
                          carry=True)
    nb.code(r'''
        # 5.8
        full = build_agent(guarded=True)
        config = run_config("full-1")
        trace = stream_trace(full, SUMMARY, config, VICTIM)
        while trace.interrupts:                                    # if the model still tries save_note: reject it
            n = len(trace.interrupts[0].value["action_requests"])
            trace = stream_trace(full, Command(resume={"decisions": [{"type": "reject"}] * n}), config, VICTIM)
        injection_report(trace, STORE, "reader-3")
        print("passed every guard (safe to cache):", answer_passed_guards(trace.messages[-1]))
    ''')

    # -- Section 6 -------------------------------------------------------------------------
    nb.section(6, "Caching", "Four places to cache. The hard part is always the key: anything that could change the "
                             "answer has to be in it.", preamble=PRECOMPUTED_SWITCH)
    nb.md("### 6.1 No cache: the same question twice")
    nb.anchor("6.1").code(r'''
        RESULTS_TEXT = section_text("results", "tinycoder")

        def summarize(chat_model, ask: str):
            """One fixed step: summarize a section. Returns (reply, milliseconds)."""
            start = time.perf_counter()
            reply = chat_model.invoke([{"role": "system", "content": AGENT_SYSTEM_V1},
                                       {"role": "user", "content": f"{RESULTS_TEXT}\n\n{ask} in two sentences."}])
            return reply, (time.perf_counter() - start) * 1000

        def show_call(ask, reply, ms):
            hit = (reply.usage_metadata or {}).get("total_cost") == 0      # LangChain's mark on a cache hit
            usage = "HIT : nothing sent, 0 tokens billed" if hit else \
                f"sent: {reply.usage_metadata['input_tokens']} in + {reply.usage_metadata['output_tokens']} out tokens, ${call_cost(reply):.6f}"
            print(f"{ask!r:34} {ms:8.1f} ms   {usage}")
    ''', carry=True)
    nb.code(r'''
        # 6.1
        for _ in range(2):
            show_call("Summarize the results section", *summarize(model, "Summarize the results section"))
        print("\nSame time, same cost, twice.")
    ''')
    nb.md("### 6.2 Exact-match cache\n\n`cache=InMemoryCache()` on this model only. The key is the full prompt plus the "
          "model settings. Then one letter changes: *Summarise*.")
    nb.anchor("6.2").code(r'''
        # 6.2: per model, so caching never leaks into other cells (set_llm_cache() would cache every model).
        # Single machine, survives restarts: langchain_community.cache.SQLiteCache(database_path=".langchain.db").
        # Shared by many servers: Redis; tomorrow the LiteLLM gateway does that for us.
        cached_model = make_chat_model(cache=InMemoryCache())
        for ask in ("Summarize the results section", "Summarize the results section", "Summarise the results section"):
            show_call(ask, *summarize(cached_model, ask))
    ''')
    nb.md("""
        ### 6.3 Tool-result cache, and getting the key right. ✏️ TODO 3 (one line)

        A second (fictional) paper, QuickEmbed. The naive cache keys search results on the query text alone. Run the
        same question on TinyCoder, then on QuickEmbed.
    """)
    nb.anchor("6.3").code('load_paper(SECOND_PDF, "quickembed")\n\n\n'
                          + defs(keys, "normalize", "make_key") + "\n\n\n"
                          + defs(tool_cache, "naive_key", "cached_search"), carry=True)
    nb.code(r'''
        # 6.3: the naive key
        naive_cache = {}
        for paper_id in ("tinycoder", "quickembed"):
            hits, hit = cached_search(SAME_Q, paper_id, naive_key, cache=naive_cache)
            print(f"{paper_id:10} {'HIT ' if hit else 'MISS'} {[h['id'] for h in hits]}  {' '.join(hits[0]['text'].split())[:60]!r}")
        print("\n⚠️ QuickEmbed got TinyCoder's chunks: a wrong answer with confident, correct-looking citations.")
    ''')
    answer3 = "key = make_key(paper_hash, EMBED_MODEL, k, normalize(query))"
    sol3 = defs(tool_cache, "search_key")
    nb.todo(stub=stub(sol3, answer3, "# ✏️ TODO 3: build the key from everything that changes the result: the paper's\n"
                                     "#   content hash, the embedding model, k and the normalized query (use make_key).\n"
                                     "key = normalize(query)   # the query alone: replace this"),
            solution=sol3)
    nb.md("### 6.3 Test: offline, with a fake search that records what it was asked")
    nb.todo_test(r'''
        # 6.3 test
        for pdf, pid in ((PDF, "tinycoder"), (SECOND_PDF, "quickembed")):
            if pid not in PAPERS:
                register_paper(pdf, pid)                              # the paper's hash; no API call
        searched = []
        def fake_search(query, paper_id, k):
            searched.append(paper_id)
            return [{"id": f"{paper_id}-c1", "text": f"a chunk of {paper_id}"}]
        test_cache = {}
        a, _ = cached_search(SAME_Q, "tinycoder", search_key, cache=test_cache, search_fn=fake_search)
        b, _ = cached_search(SAME_Q, "quickembed", search_key, cache=test_cache, search_fn=fake_search)
        c, hit = cached_search("what is the MAIN result", "quickembed", search_key, cache=test_cache, search_fn=fake_search)
        assert a != b, "TODO 3: two papers share one cache entry: put paper_hash in the key"
        assert hit and c == b and searched == ["tinycoder", "quickembed"], "TODO 3: a repeated question should be a hit"
        assert search_key("q", "h", 4) != search_key("q", "h", 8), "TODO 3: k changes the result: put k in the key"
        saved_model, EMBED_MODEL = EMBED_MODEL, "another-embedding-model"
        other = search_key("q", "h", 4)
        EMBED_MODEL = saved_model
        assert other != search_key("q", "h", 4), "TODO 3: the embedding model changes the result: put EMBED_MODEL in the key"
        print("✅ TODO 3: one entry per paper, model, k and question; a repeat is a hit")
    ''')
    nb.rescue(sol3, f"`{answer3}`")
    nb.code(r'''
        # 6.3: the right key
        tool_cache_ = {}
        for paper_id in ("tinycoder", "quickembed", "quickembed"):
            hits, hit = cached_search(SAME_Q, paper_id, search_key, cache=tool_cache_)
            print(f"{paper_id:10} {'HIT ' if hit else 'MISS'} {[h['id'] for h in hits]}  {' '.join(hits[0]['text'].split())[:60]!r}")
        print("\nTwo papers, two results; the repeat on QuickEmbed is a hit.")
    ''')
    nb.md("### 6.4 Semantic cache\n\nEmbed the question, find the closest cached question for this paper, reuse its "
          "answer above a threshold. We calibrate the threshold with **your** embedding model: a reworded question that "
          "should hit, and a near-miss (pass@10) that must not.")
    nb.anchor("6.4").code(defs(semantic, "Lookup", "SemanticCache"), carry=True)
    nb.code(r'''
        # 6.4
        semantic = SemanticCache(lambda texts: day2.embed_texts(texts), threshold=1.0, scope="tinycoder")
        trace = stream_trace(build_agent(guarded=True), PASS1, run_config("semantic-1"), Context(user_id="reader-6"),
                             show=False)
        semantic.put(PASS1, trace.answer, passed_guards=answer_passed_guards(trace.messages[-1]))
        print(f"cached: {PASS1!r} → {trace.answer[:90]}\n")
        scores = {q: semantic.lookup(q).score for q in (REWORDED, NEAR_MISS)}
        for q, s in scores.items():
            print(f"  similarity {s:.3f}  {q}")
        LOOSE = round(min(scores.values()) - 0.05, 2)
        RECOMMENDED = round((scores[REWORDED] + scores[NEAR_MISS]) / 2, 2) if scores[REWORDED] > scores[NEAR_MISS] else None
        for name, threshold in (("loose", LOOSE), ("recommended", RECOMMENDED)):
            if threshold is None:
                print(f"\n{EMBED_MODEL} scores the near-miss at least as high as the rewording: no threshold separates "
                      "them, so don't use a semantic cache for questions with numbers on this model.")
                continue
            semantic.threshold = threshold
            print(f"\n{name} threshold {threshold:.2f}:")
            for q in (REWORDED, NEAR_MISS):
                found = semantic.lookup(q)
                print(f"  {'HIT ' if found.answer else 'MISS'} {q}" + (f"  → reuses the pass@1 answer" if found.answer else ""))
        print("\nNever cache an answer that failed a guard:",
              "stored" if semantic.put("Who are the authors?", CITATION_FAILED, passed_guards=False) else "refused")
    ''')
    nb.md("### 6.5 Provider prompt caching\n\nTwo different questions behind the same long prefix: the system prompt, "
          "then the whole paper. On providers that support it, part of the second call's input shows up as cached "
          "tokens. A zero, or no report at all, is a finding, not a bug.")
    nb.anchor("6.5").code(defs(prompt_cache, "cached_input_tokens"), carry=True)
    nb.code(r'''
        # 6.5: stable part first, question last
        PREFIX = AGENT_SYSTEM_V1 + "\n\nThe whole paper:\n" + "\n\n".join(c["text"] for c in PAPERS["tinycoder"]["chunks"])
        for question in ("What is the main result?", "What are the limitations?"):
            reply = model.invoke([{"role": "system", "content": PREFIX}, {"role": "user", "content": question}])
            cached = cached_input_tokens(reply)
            print(f"{question:28} input {reply.usage_metadata['input_tokens']:>6,} tokens; cached: "
                  + ("your provider didn't report cached tokens" if cached is None else f"{cached:,}"))
    ''')

    # -- Closing ---------------------------------------------------------------------------
    nb.md(f"""
        ---
        # Production Cheatsheet: agents, tools, safety and caching

        **Tools**
        - The docstring is the spec: what, when, and what it returns ({nb.link("3.2")}, {nb.link("3.4")})
        - Validate arguments; the model wrote them ({nb.link("3.3")})
        - Return errors the model can act on; don't raise ({nb.link("3.5")})
        - Read-only by default; writes are the exception ({nb.link("4.4")})

        **Agents**
        - Use a workflow when the steps are fixed
        - Read the trace: steps, tokens, cost ({nb.link("4.1")}, {nb.link("4.2")})
        - Cap model calls and tool calls on every run ({nb.link("4.5")}, {nb.link("5.2")})

        **Memory**
        - Thread ID for the conversation, store for the reader ({nb.link("4.3")}, {nb.link("4.4")})
        - A person approves every memory write ({nb.link("5.6")})
        - Never store what a document told you about the reader ({nb.link("5.4")})

        **Safety**
        - Tool output is untrusted data, never instructions ({nb.link("5.5")})
        - Delimit document text and flag instruction-like content ({nb.link("5.5")})
        - Redact personal data before the model sees it ({nb.link("5.3")})
        - Check citations against what was retrieved in this run ({nb.link("5.7")})
        - Least privilege: no tool the task doesn't need ({nb.link("5.8")})

        **Caching**
        - Exact-match first; semantic only with a high threshold and near-miss tests ({nb.link("6.2")}, {nb.link("6.4")})
        - Key = everything that changes the answer: paper, model, prompt version, user ({nb.link("6.3")})
        - Never cache an answer that failed a guard ({nb.link("6.4")})
        - Stable prefix first, question last ({nb.link("6.5")})
    """)
    nb.md("""
        ---
        ### Extensions (homework)

        **E1. Five injections.** Write three to five new hidden instructions (a different phrasing, a tiny font, an
        instruction split across two lines) in a copy of the paper, and count how many get past `find_injection`.
        The best ones go into tomorrow's evaluation set.

        **E2. A cost budget.** `paper_agent/agent/guards.py` has `cost_budget(budget_usd)`, an `@after_model`
        middleware that stops a run when its spend (Day 1 prices) passes a budget. Add it to `guard_stack()` and rerun
        4.5 with a budget of $0.002.

        **E3. Summarize long conversations.** Add `SummarizationMiddleware(model, trigger=("messages", 10))` and push a
        conversation past the trigger. What does the model see afterwards?

        **E4. MCP.** Serve `search_paper` and `get_section` over MCP, so any MCP client can use them. What changes in
        the tools themselves? (Nothing.)

        ---
        ### Tomorrow (Day 4): ship it

        We wrap the agent in a **FastAPI** service and put the **LiteLLM gateway** in front of the model: per-user
        budgets and rate limits, today's cache in the gateway's Redis, tracing for every step, and an evaluation set
        that runs whenever a prompt or a model changes. Tonight: finish any section you missed with its catch-up cell,
        and try one extension.
    """)
    return nb


# ===========================================================================
# Demo (instructor): D2.2 and D2.6
# ===========================================================================
DEMO_SETUP = FETCH + INSTALL + r'''
NAMES = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
try:
    from google.colab import userdata
    for name in NAMES:
        try:
            if userdata.get(name):
                os.environ[name] = userdata.get(name)
        except Exception:
            pass
except ImportError:
    pass
from openai import OpenAI
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], max_retries=2, timeout=30)
MODEL = os.environ["LLM_MODEL"].split(",")[0].strip()
print("✅ ready:", MODEL)
'''


def demo() -> Notebook:
    nb = Notebook("day3-demo")
    nb.md("""
        # Day 3 Instructor Demo: tool calling and caching (D2.2, D2.6)

        **Run by the instructor only** (Section 2); shared with participants afterwards. Same Secrets as the
        code-along. ⚠️ Never put company data into free APIs. TinyCoder is fictional.

        No LangChain here on purpose: the plain `openai` SDK shows the mechanism under every agent framework.
    """)
    nb.code("# Setup: installs today's libraries if needed and connects to your provider\n" + clean(DEMO_SETUP))
    nb.md("---\n## D2.2 How tool calling works · *Slide 2.2*\n\nOne request with a `tools` list. The reply has no text: "
          "just a tool name and arguments. Nothing has run.")
    nb.code(r'''
        # D2.2: the raw request
        tools = [{"type": "function", "function": {
            "name": "search_paper",
            "description": "Search the current paper for the passages most relevant to a question. "
                           "Use it for anything about what the paper says, measures or claims.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                            "k": {"type": "integer", "default": 4}},
                           "required": ["query"]}}}]
        request = {"model": MODEL, "temperature": 0, "tools": tools,
                   "messages": [{"role": "user", "content": "How does TinyCoder do on long files?"}]}
        print(json.dumps(request, indent=2))
    ''')
    nb.code(r'''
        # D2.2: the raw reply
        response = client.chat.completions.create(**request)
        print(json.dumps(response.model_dump(exclude_none=True), indent=2))
        call = response.choices[0].message.tool_calls[0]
        print(f"\nThe model asks for {call.function.name}({call.function.arguments}). Our code runs it and sends the "
              "result back as a 'tool' message.")
    ''')
    nb.md("---\n## D2.6 Four places to cache · *Slide 2.6*\n\nAn exact-match cache in five lines: the key is the whole "
          "request. The same call twice, then a one-word change.")
    nb.code(r'''
        # D2.6
        import hashlib
        CACHE = {}

        def cached_call(question):
            body = {"model": MODEL, "temperature": 0, "max_tokens": 200,
                    "messages": [{"role": "user", "content": question}]}
            key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()   # the whole request
            start = time.perf_counter()
            hit = key in CACHE
            if not hit:
                CACHE[key] = client.chat.completions.create(**body)
            return CACHE[key], hit, (time.perf_counter() - start) * 1000

        for question in ("What is sliding-window attention?", "What is sliding-window attention?",
                         "What is sliding-window attention exactly?"):
            reply, hit, ms = cached_call(question)
            print(f"{'HIT ' if hit else 'MISS'} {ms:8.1f} ms  {question!r}")
    ''')
    return nb
