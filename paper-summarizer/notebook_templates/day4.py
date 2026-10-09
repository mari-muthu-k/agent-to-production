"""Day 4 notebooks: serve, control, observe, evaluate. Follows reference/Day4_Instructor_Guide.docx cell for cell.

Code-along (Day4_CodeAlong.ipynb)          Sections 4-7, cells 0.1-7.5 and C.1-C.2, three TODOs (4.3, 5.3, 7.4)
Instructor demo (Day4_Instructor_Demo)     D0-DX: the production version (LiteLLM proxy + Postgres + Redis + api)

Teaching cells show package code with defs()/consts(), so the notebook runs exactly the code the tests run.
In Colab, 0.1 installs requirements-day4.txt and 4.0 downloads the course package (Days 2-4) from GitHub;
in Docker the same files are copied from the repo checkout (COURSE_REPO).
"""
from pathlib import Path

from nbgen import Notebook, clean, consts, defs

from paper_agent.evals import gate, metrics, runner
from paper_agent.service import app as service
from paper_agent.service import usage, validate

NAMES = ("Day4_CodeAlong.ipynb", "Day4_Instructor_Demo.ipynb")
ROOT = Path(__file__).resolve().parents[1]
REPO = "https://raw.githubusercontent.com/mari-muthu-k/agent-to-production/main"
PKG = "paper-summarizer/paper_agent"
MODULES = ["__init__.py", "rag/__init__.py", "rag/pdf_rag.py", "agent/__init__.py", "agent/papers.py", "agent/tools.py",
           "agent/guards.py", "agent/memory.py", "agent/trace.py", "agent/model.py", "agent/policy.py", "agent/fakes.py",
           "prompts/__init__.py", "prompts/agent_v1.txt", "prompts/agent_v2.txt", "prompts/judge_v1.txt",
           "fixtures/__init__.py", "fixtures/tinycoder_pdf.py", "evals/__init__.py", "evals/metrics.py",
           "evals/runner.py", "evals/gate.py", "evals/golden_tinycoder.json",
           *[f"service/{m}.py" for m in ("__init__", "config", "errors", "gateway", "usage", "store", "pipeline",
                                         "validate", "app", "tracing", "keys")]]
FILES = {"requirements-day4.txt": "paper-summarizer/requirements-day4.txt",
         **{f"paper_agent/{m}": f"{PKG}/{m}" for m in MODULES}}

FETCH = rf'''
import importlib, importlib.metadata, json, os, shutil, sys, time, urllib.request, warnings
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
# 0.1 INSTALL: today's libraries, pinned (requirements-day4.txt). Run it once, before 2:00 if you can.
fetch("requirements-day4.txt", force=True)
PINS = dict(line.split("==") for line in Path("requirements-day4.txt").read_text().split() if "==" in line)

def installed(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

wrong = {p: installed(p) for p, v in PINS.items() if installed(p) != v}
IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
if wrong and not IN_COLAB:                                    # never pip-install into someone's own Python
    raise RuntimeError(f"Not in Colab, and these differ from requirements-day4.txt: {wrong}. "
                       "Install them in this environment yourself (pip install -r requirements-day4.txt).")
if wrong:
    print(f"Installing {len(wrong)} pinned package(s): a minute or two the first time ...")
    !pip install -q -r requirements-day4.txt
    loaded = [m for m in ("pydantic", "fastapi", "httpx", "openai", "langchain_core", "opentelemetry") if m in sys.modules]
    if loaded:
        print("\n" + "=" * 72 + "\n✅ Installed. ⚠️  Now RESTART THE SESSION (Runtime → Restart session), then run 4.0.\n"
              f"Older versions of {', '.join(loaded)} were already loaded in this runtime.\n" + "=" * 72)
        raise SystemExit("restart the session, then run 4.0")
print("✅ today's libraries are installed: run 4.0")
'''

SETUP = (r'''
# 4.0 SETUP and CATCH-UP: Days 1-3 in one cell. Downloads the course code, connects to your provider through
# LiteLLM, draws the TinyCoder PDF, parses, chunks, embeds and indexes it, and builds the agent with its tools
# and citation check. Safe to run again at any time. You don't need to read it today: you wrote all of it.
''' + clean(FETCH) + r'''
for name in FILES:
    fetch(name)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")      # LiteLLM: no price download at import
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
for noisy in ("pypdf", "chromadb", "LiteLLM", "LiteLLM Router", "httpx", "uvicorn.error"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# Your provider, from Colab Secrets (🔑, notebook access ON) or environment variables. Table at the top.
# ⚠️ Never put company data into free APIs.
from paper_agent.service import config
config.load_secrets()
missing = [n for n in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "EMBED_MODEL") if not os.environ.get(n)]
assert not missing, f"Missing {missing}: add them in Colab Secrets (🔑) and turn on notebook access, then run 4.0 again."

import queue, re, threading, uuid
from typing import Callable, Optional
import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from langchain.agents.middleware import before_model
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from paper_agent.agent.fakes import FakeToolModel                        # offline model for the TODO tests
from paper_agent.agent.guards import CITATION
from paper_agent.agent.papers import PAPERS
from paper_agent.fixtures.tinycoder_pdf import make_pdf                  # Day 1's paper as a real 3-page PDF
from paper_agent.prompts import load_prompt
from paper_agent.service.config import ALIAS, DEFAULT_BUDGET_USD, MAX_TOKENS, MAX_UPLOAD_MB
from paper_agent.service.errors import BudgetExceededError, RateLimitedError
from paper_agent.service.gateway import CURRENT_KEY, make_chat_model, make_router
from paper_agent.service.pipeline import EXPLAIN_Q, Ctx, ask_paper, build_agent, explain_events
from paper_agent.service.store import PaperStore, setup_retrieval
import paper_agent.service.store as paper_store                          # search(): the `retrieve` span (6.3)
from paper_agent.service.tracing import TRACER, setup_tracing, show_trace
from paper_agent.service.usage import (BUDGETS, CALLS, SPEND, message_cost, record_usage, trace_model_call,
                                       trace_tool_call, usage_of)
from paper_agent.service.app import (API_KEYS, ERROR_LOG, KEY_USERS, LOG, SERVERS, Question, LogCapture,
                                     attach_log_handler, check_rate_limit, error_body, extra_middleware, key_info,
                                     set_rate_limit)
from paper_agent.evals.runner import EVAL_PAUSE_S, baseline_record, chunk_index, golden_set
from paper_agent.evals.gate import compare

# Days 1-3, rebuilt: the gateway (LiteLLM Router), retrieval (Day 2) and the agent (Day 3)
router = make_router()                       # alias paper-explainer -> your model; fallback, retries, timeout (5.1)
setup_retrieval(router, ".")                 # Day 2's embeddings go through the gateway too (alias paper-embed)
PDF_BYTES = make_pdf("tinycoder.pdf")        # FICTIONAL paper, drawn with reportlab: same bytes every time
papers = PaperStore(".")
PAPER_ID, _ = papers.ingest(PDF_BYTES)       # validate, parse, chunk, embed, index (Days 2-3)
agent = build_agent(router=router)           # Day 3's agent: search_paper + get_section, guards, citation check
ALICE, BOB = {"Authorization": "Bearer key-alice"}, {"Authorization": "Bearer key-bob"}
Q_SPEED = "How much faster is TinyCoder than the 7B baseline?"
print(f"✅ paper {PAPER_ID}: {len(papers.get(PAPER_ID)['chunks'])} chunks, indexed · model alias {ALIAS} → "
      f"{os.environ['LLM_MODEL'].split(',')[0]} · agent ready")
''')


def stub(solution: str, answer: str, todo: str) -> str:
    """The TODO version of a function: its answer replaced by a TODO comment (and `pass`)."""
    if answer not in solution:
        raise ValueError(f"answer not found: {answer!r}")
    at = solution.index(answer)
    indent = solution[solution.rfind("\n", 0, at) + 1:at]
    return solution.replace(answer, todo.replace("\n", "\n" + indent))


# ===========================================================================
# Code-along
# ===========================================================================
def practice() -> Notebook:
    nb = Notebook("day4-codealong", setup=SETUP)
    nb.md("""
        # Day 4 Code-Along: ship it (serve, control, observe, evaluate)

        **Hands-on AI Workshop · Day 4 of 4 · Capstone: Research Paper Summarizer Agent with Memory**

        Today the agent becomes a service two hundred colleagues could use at once. Cell numbers (4.2, 5.4 …) match
        the tags on the slides.

        | Section | What you build | Failure shown on purpose |
        |---|---|---|
        | **4. Serve it** | FastAPI: upload, ask, stream, usage; API-key auth; TODO 1: validate uploads | A text file pretending to be a PDF; an 11 MB upload |
        | **5. Control** | LiteLLM Router: alias, retries, fallback; per-user budgets (TODO 2) and rate limits | Broken primary model; Bob runs out of budget; Alice hits the rate limit |
        | **6. Observe** | One JSON log line per request; an OpenTelemetry trace tree | A slow vector search, found from the trace |
        | **7. Evaluate** | Golden set, rule scorers, LLM judge, baseline.json; TODO 3: the regression gate | A friendlier prompt silently drops citations and is blocked |

        **Fell behind, or your runtime restarted?** Run the **⏩ catch-up cell** (5.0, 6.0, 7.0) at the start of the
        section you want to rejoin. **A cell failed?** Post the error in the Q&A channel; a TA will help.

        ### Your provider: Colab Secrets (🔑 in the left sidebar, notebook access ON)

        | Secret | What it is |
        |---|---|
        | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | The same as Days 1-3 (pinned in the channel). The model must support **tool calling** |
        | `EMBED_MODEL` | The embedding model, as on Days 2-3 |
        | `LLM_FALLBACK_MODEL` | *Optional.* A second model for 5.2; without it the fallback is your primary again |
        | `PRICE_IN_PER_M`, `PRICE_OUT_PER_M` | *Optional.* USD per 1M tokens; leave them out for the workshop prices ($0.15 / $0.60) |

        > ⚠️ **Never put company data into free APIs.**
        >
        > 📄 **TinyCoder, its authors and its numbers are FICTIONAL**, written for this workshop. Cell 4.0 draws it as a
        > three-page PDF.

        > **Production vs Colab, honestly:** in production LiteLLM runs as a **proxy** with Postgres and Redis, and owns
        > budgets and rate limits for every app. Colab can't run Postgres for 200 people, so here LiteLLM runs as a
        > **library** inside the service, and budgets and limits are a few lines of our own code. Same mechanism: check
        > before every call, record after.
    """)
    nb.md("### 0.1 Install\n\nInstalls today's libraries (pinned: `requirements-day4.txt`). If it asks you to "
          "**restart the session**, do it (Runtime → Restart session), then run 4.0.")
    nb.code(clean(FETCH) + "\n" + clean(INSTALL))

    # -- Section 4 -------------------------------------------------------------------------
    nb.section(4, "Serve it", "FastAPI serves the agent from inside this notebook: uvicorn runs in a background "
                              "thread, and we talk to it over HTTP like any other client would.", catch_up=False)
    nb.md("### 4.0 Setup and catch-up: Days 1-3 in one cell\n\nThe TinyCoder paper as a real PDF, parsing, chunks, "
          "retrieval, the LangChain agent and the citation check. It is also the catch-up cell for Section 4.")
    nb.code(SETUP)
    nb.md("### 4.1 Setup check\n\nOne question through the agent: the answer, its citations, how many LLM calls it took, "
          "the tokens and the cost. When you see **ready**, put a green tick in the chat.")
    nb.anchor("4.1").code(r'''
        # 4.1
        result = ask_paper(PAPER_ID, Q_SPEED, "setup-check", agent)
        print("answer   :", result["answer"])
        print("citations:", result["citations"], "valid" if result["citations_valid"] else "NOT VALID")
        print(f"LLM calls: {result['llm_calls']} · tokens: {result['input_tokens']:,} in + {result['output_tokens']:,} out"
              f" · cost: ${result['cost_usd']:.6f} · latency: {result['latency_ms']:,} ms")
        print("✅ ready" if result["citations_valid"] and result["llm_calls"] else "⚠️ something is off: post this output in the channel")
    ''')
    nb.md("""
        ### 4.2 The service

        The whole service. Four endpoints: `POST /papers` (upload), `POST /papers/{id}/ask`, `POST /papers/{id}/explain`
        (streamed) and `GET /me/usage`. `current_user` maps every API key to a user (at work: your single sign-on).
        `error_response` turns our errors into clear HTTP answers. The `request_log` middleware writes one line per
        request: silent for now, switched on in Section 6.
    """)
    nb.anchor("4.2").code(defs(service, "current_user", "error_response", "build_app")
                          + "\n\n\napp = build_app(agent, papers)", carry=True)
    nb.code(r'''
        # 4.2: the four routes
        for route in app.routes:
            if route.path.startswith(("/papers", "/me")):
                print(f"{', '.join(sorted(route.methods)):5} {route.path}")
    ''')
    nb.md("""
        ### 4.3 Your turn: never trust the file. ✏️ TODO 1 (two lines)

        The filename and the `Content-Type` both come from the client. The bytes don't lie: every PDF starts with
        `%PDF-`. **TODO 1:** (1) if `data` doesn't start with `b"%PDF-"`, raise `HTTPException(415, ...)`; (2) if it is
        bigger than `MAX_UPLOAD_MB`, raise `HTTPException(413, ...)`. Check before parsing: a huge or broken file is
        exactly the work we want to avoid.
    """)
    answer1 = ('if not data.startswith(b"%PDF-"): raise HTTPException(415, "Only PDF files are accepted.")\n'
               '    if len(data) > MAX_UPLOAD_MB * 1024 * 1024: raise HTTPException(413, f"Max upload size is '
               '{MAX_UPLOAD_MB} MB.")')
    sol1 = defs(validate, "validate_upload")
    nb.anchor("4.3").todo(stub=stub(sol1, answer1, "# ✏️ TODO 1 (two lines):\n"
                                                   "#   1. not starting with b\"%PDF-\"  -> raise HTTPException(415, \"Only PDF files are accepted.\")\n"
                                                   "#   2. len(data) > MAX_UPLOAD_MB * 1024 * 1024 -> raise HTTPException(413, ...)\n"
                                                   "pass"),
                          solution=sol1)
    nb.md("### 4.3a Test: offline, no server and no LLM call\n\nA text file named `.pdf`, an 11 MB upload and a wrong "
          "API key. You want **415, 413 and 401**.")
    nb.todo_test(r'''
        # 4.3a test
        test_client = TestClient(app, raise_server_exceptions=False)
        fake_pdf = test_client.post("/papers", headers=ALICE, files={"file": ("notes.pdf", b"Just some notes, not a PDF.", "application/pdf")})
        too_big = test_client.post("/papers", headers=ALICE, files={"file": ("huge.pdf", b"%PDF-1.4\n" + b"0" * (11 * 1024 * 1024), "application/pdf")})
        wrong_key = test_client.post("/papers", headers={"Authorization": "Bearer key-mallory"},
                                     files={"file": ("tinycoder.pdf", PDF_BYTES, "application/pdf")})
        for label, r in (("text file named .pdf", fake_pdf), ("11 MB upload", too_big), ("wrong API key", wrong_key)):
            print(f"{label:22} → {r.status_code} {r.json()}")
        assert fake_pdf.status_code == 415, "TODO 1: check the first bytes: data.startswith(b'%PDF-'), else HTTPException(415, ...)"
        assert too_big.status_code == 413, "TODO 1: check the size: len(data) > MAX_UPLOAD_MB * 1024 * 1024 -> HTTPException(413, ...)"
        assert wrong_key.status_code == 401
        print("\n✅ TODO 1 passed: 415, 413 and 401, before any parsing")
    ''')
    nb.rescue(sol1, "Two lines, in this order:\n\n```\n" + clean(answer1.replace("\n    ", "\n")) + "\n```")
    nb.md("""
        ### 4.4 Start the server and upload a paper

        The server runs in the background of this notebook (port 8765), and we talk to it over HTTP. The same file
        twice, under two names: the paper ID is a hash of the **content**, not the name, so it is parsed and embedded
        once, whoever uploads it. (4.0 already processed it, so both say so.)
    """)
    nb.anchor("4.4").code(defs(service, "start_server") + "\n\n\nBASE = start_server(app)\n"
                          "http = httpx.Client(base_url=BASE, timeout=120, trust_env=False)", carry=True)
    nb.code(r'''
        # 4.4
        for name in ("tinycoder.pdf", "tinycoder_FINAL_v2.pdf"):
            r = http.post("/papers", headers=ALICE, files={"file": (name, PDF_BYTES, "application/pdf")})
            print(f"{name:24} → {r.status_code} {r.json()['paper_id']}  {r.json()['status']}")
    ''')
    nb.md("### 4.5 Ask a question over HTTP\n\nJSON in, JSON out. Besides the answer: citations and whether they are "
          "valid, LLM calls, tokens, cost, and which model answered. And in the headers, a request ID: keep it for "
          "Section 6.")
    nb.anchor("4.5").code(r'''
        # 4.5
        r = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": Q_SPEED})
        body = r.json()
        print("X-Request-ID:", r.headers["X-Request-ID"], "| status", r.status_code)
        print(json.dumps({k: body[k] for k in ("answer", "citations", "citations_valid", "llm_calls", "input_tokens",
                                               "output_tokens", "cost_usd", "answered_by")}, indent=1))
        LAST_COST = body["cost_usd"]                       # what one question costs: 5.4 uses it
    ''')
    nb.md("### 4.6 Streaming progress\n\nServer-sent events: a plain HTTP response that stays open and sends lines as "
          "they are ready. Compare the two numbers at the bottom.")
    nb.anchor("4.6").code(r'''
        # 4.6
        start, first, full = time.perf_counter(), None, None
        with http.stream("POST", f"/papers/{PAPER_ID}/explain", headers=ALICE) as r:
            event = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    t = time.perf_counter() - start
                    first = first or t
                    data = json.loads(line[6:])
                    if event == "answer":
                        full = t
                    shown = {"progress": lambda d: " ".join(str(v) for k, v in d.items()),
                             "answer": lambda d: d["answer"][:90] + " …",
                             "done": lambda d: f"{d['llm_calls']} LLM calls, ${d['cost_usd']:.6f}"}.get(event, str)(data)
                    print(f"{t:5.1f}s  {event:9} {shown}")
        print(f"\nfirst event after {first:.1f}s · full answer after {full:.1f}s")
    ''')

    # -- Section 5 -------------------------------------------------------------------------
    nb.section(5, "Control cost and failures", "A gateway in front of every model call, a budget per user checked "
                                               "before every call, and a rate limit per user.")
    nb.md("### 5.1 Open up the gateway\n\nYou've been using LiteLLM since 4.0. Our code only ever asks for the alias "
          "`paper-explainer`; the Router maps it to a real model, and knows the fallback, the retries and the timeout.")
    nb.anchor("5.1").code(r'''
        # 5.1: the Router behind every model call (keys hidden)
        for d in router.model_list:
            p = d["litellm_params"]
            print(f"{d['model_name']:26} → {p['model']:34} (deployment id: {d['model_info']['id']})")
        print("\nfallbacks  :", router.fallbacks)
        print("num_retries:", router.num_retries, "| timeout:", router.timeout, "s")
        chat = make_chat_model(router)
        print(f"LangChain  : {type(chat).__name__}(model_name={chat.model_name!r}, max_tokens={chat.max_tokens}, "
              f"max_retries={chat.max_retries})  ← one attempt, no retries")
    ''')
    nb.md("""
        | Day 1 `llm_client.py` (by hand) | Day 4 (one home each) |
        |---|---|
        | Retries + backoff | LiteLLM `num_retries` |
        | Timeout | LiteLLM `timeout` |
        | Fallback model | LiteLLM `fallbacks` |
        | `max_tokens` | `max_tokens` on every model call (`MAX_TOKENS = 400`) |
        | Budget guard | per user, before every model call (5.3) |
        | Log line | one per request, plus a trace (6.x) |

        LangChain's `max_retries=1` means **one attempt**: one retry layer, the Day 1 rule.
    """)
    nb.md("### 5.2 Force a fallback\n\nPoint the primary at a model that doesn't exist and ask through the server. "
          "No restart: the next request just uses the new agent. The cell puts the real primary back at the end.")
    nb.anchor("5.2").code(r'''
        # 5.2
        app.state.agent = build_agent(router=make_router(primary="this-model-does-not-exist"))
        r = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": Q_SPEED})
        print("status     :", r.status_code)
        print("answered_by:", r.json().get("answered_by"), "← the primary failed, the fallback answered")
        print("answer     :", r.json().get("answer", r.json())[:100], "…")
        app.state.agent = agent                              # the real primary again
        print("\nprimary restored:", http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE,
                                                json={"question": Q_SPEED}).json()["answered_by"])
    ''')
    nb.md("""
        ### 5.3 Your turn: a budget on every call. ✏️ TODO 2 (two lines)

        Check once per question and one question can still run four model calls straight past the budget. So
        `budget_guard` is a `@before_model` middleware: LangChain runs it before **every** model call. `record_usage`
        (written in 4.0) adds each call's cost to `SPEND` after it. **TODO 2:** inside `check_budget`, if the user has
        spent their whole budget, raise `BudgetExceededError(user_id, spent, budget)`.
    """)
    answer2 = "if spent >= budget:\n        raise BudgetExceededError(user_id, spent, budget)"
    sol2 = defs(usage, "check_budget", "budget_guard")
    nb.anchor("5.3").todo(stub=stub(sol2, answer2, "# ✏️ TODO 2 (two lines): if spent has reached the budget,\n"
                                                   "#   raise BudgetExceededError(user_id, spent, budget)\n"
                                                   "pass"),
                          solution=sol2)
    nb.md("### 5.3a Test: offline, with a fake model\n\nUnder the budget passes, at or over raises, and an agent run "
          "stops **before** its next model call once the spend crosses the budget.")
    nb.todo_test(r'''
        # 5.3a test
        BUDGETS["test-user"], SPEND["test-user"] = 0.001, 0.0009
        check_budget("test-user")                                       # under: fine
        for spent in (0.001, 0.002):
            SPEND["test-user"] = spent
            try:
                check_budget("test-user")
                raise AssertionError(f"TODO 2: spent ${spent} of a $0.001 budget should raise BudgetExceededError")
            except BudgetExceededError as e:
                print("raised as expected:", e)
        fake = FakeToolModel(script=[AIMessage("", tool_calls=[{"name": "get_section", "args": {"section_id": "results"}, "id": "t1"}]),
                                     AIMessage("", tool_calls=[{"name": "get_section", "args": {"section_id": "limitations"}, "id": "t2"}]),
                                     AIMessage("Done.")])
        BUDGETS["test-user"], SPEND["test-user"] = 1e-9, 0.0             # the first call spends more than this
        try:
            build_agent(model=fake, extra_middleware=[budget_guard]).invoke(
                {"messages": [{"role": "user", "content": "Tell me everything."}]}, context=Ctx("test-user", PAPER_ID))
        except BudgetExceededError:
            pass
        print(f"model calls made: {len(fake.requests)} (the script had 3)")
        assert len(fake.requests) == 1, "TODO 2: budget_guard should stop the run before the second model call"
        print("✅ TODO 2 passed: checked before every model call")
    ''')
    nb.rescue(sol2, "Two lines, where the TODO is:\n\n```\nif spent >= budget:\n    raise BudgetExceededError(user_id, spent, budget)\n```")
    nb.md("### 5.4 Two users, two budgets\n\nThe guard joins the live agent. Alice gets a dollar; Bob gets about one and "
          "a half questions' worth, based on what the last question actually cost. Watch Bob, then the last line.")
    nb.anchor("5.4").code(r'''
        agent = build_agent(router=router, extra_middleware=[budget_guard])    # 5.4: the guard joins the live agent
        app.state.agent = agent
        BUDGETS["alice"] = 1.00
    ''', carry=True)
    nb.code(r'''
        # 5.4
        if "LAST_COST" not in globals():                    # joined late: measure one question now
            LAST_COST = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": Q_SPEED}).json()["cost_usd"]
        BUDGETS["bob"] = round(1.5 * LAST_COST, 6)
        SPEND["bob"], CALLS["bob"] = 0.0, 0
        print(f"Bob's budget: ${BUDGETS['bob']:.6f} (1.5 × the last question's ${LAST_COST:.6f})\n")
        for n in (1, 2, 3):
            r = http.post(f"/papers/{PAPER_ID}/ask", headers=BOB, json={"question": Q_SPEED})
            print(f"Bob, question {n}: {r.status_code}", r.json()["message"] if r.status_code != 200 else f"(spent so far ${SPEND['bob']:.6f})")
        r = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": Q_SPEED})
        print(f"Alice           : {r.status_code} (budgets are per user)")
        over = SPEND["bob"] - BUDGETS["bob"]
        print(f"\nBob spent ${SPEND['bob']:.6f} of ${BUDGETS['bob']:.6f}: over by ${over:.6f}. The check runs before a call, "
              "the cost is known after it, so one call can overshoot. max_tokens on every call keeps that call small.")
    ''')
    nb.md("### 5.5 Rate limits\n\nThree requests per minute for Alice, then five requests. The cell removes the limit at "
          "the end. (In-process counter: fine for one Colab process; the proxy keeps it in Redis.)")
    nb.anchor("5.5").code(r'''
        # 5.5
        set_rate_limit("alice", 3)
        for n in range(1, 6):
            r = http.get("/me/usage", headers=ALICE)
            extra = f"Retry-After: {r.headers['Retry-After']}s · {r.json()['message']}" if r.status_code == 429 else ""
            print(f"request {n}: {r.status_code} {extra}")
        set_rate_limit("alice", None)                       # no limit for the rest of the notebook
    ''')
    nb.md("### 5.6 Spend per user\n\nEach user's calls, spend, budget and what's left. Bob's remaining is negative: the "
          "overshoot from 5.4. In production these numbers live in the LiteLLM proxy (slide).")
    nb.anchor("5.6").code(r'''
        # 5.6
        for user, headers in (("alice", ALICE), ("bob", BOB)):
            print(json.dumps(http.get("/me/usage", headers=headers).json()))
    ''')

    # -- Section 6 -------------------------------------------------------------------------
    nb.section(6, "Observe it", "A log line says what happened in a request; a trace says where the time went.")
    nb.md("### 6.1 One log line per request\n\nThe service has written a line for every request since 4.2; nobody was "
          "listening. Attach a handler (in production: Datadog, CloudWatch …). JSON, so a machine can search it, and "
          "no question, answer or document text in it.")
    nb.anchor("6.1").code("log = attach_log_handler(LogCapture())      # prints each line, and keeps it for searching",
                          carry=True)
    nb.code(r'''
        # 6.1
        r = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": "What attention mechanism does TinyCoder use?"})
        ticket = r.headers["X-Request-ID"]
        print(f"\n🎫 Support ticket: 'this answer was wrong', request id {ticket} (from the response header)")
        print("one search →", json.dumps(log.find(ticket)))
    ''')
    nb.md("### 6.2 Traces\n\nThe code loaded in 4.0 already marks each step with a span; they went nowhere. Switch "
          "collection on (in memory here; one line changes it to Jaeger or Datadog), ask one question, read the tree.")
    nb.anchor("6.2").code("exporter = setup_tracing(InMemorySpanExporter())      # from now on, every span is kept here",
                          carry=True)
    nb.code(r'''
        # 6.2: the span lines that were in 4.0's code all along
        import inspect, paper_agent.service.pipeline, paper_agent.service.usage
        for module in (paper_agent.service.pipeline, paper_agent.service.usage, paper_store):
            for line in inspect.getsource(module).splitlines():
                if "start_as_current_span(" in line:
                    print(f"{module.__name__.split('.')[-1]:9}│ {line.strip()}")
        exporter.clear()
        r = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": "What are the main limitations of TinyCoder?"})
        print()
        spans = show_trace(exporter)
    ''')
    nb.md("### 6.3 Find the slow step\n\nThe vector search gets a second and a half slower, as if the database were "
          "overloaded. The log line says the request was slow; the trace says why. The cell resets the delay.")
    nb.anchor("6.3").code(r'''
        # 6.3
        SEARCH_DELAY_S = 1.5
        paper_store.SEARCH_DELAY_S = SEARCH_DELAY_S
        exporter.clear()
        r = http.post(f"/papers/{PAPER_ID}/ask", headers=ALICE, json={"question": Q_SPEED})
        print(f"\n😤 user: 'it took {log.find(r.headers['X-Request-ID'])['latency_ms'] / 1000:.1f} seconds'\n")
        spans = show_trace(exporter)
        retrieve = [s for s in spans if s.name == "retrieve"]
        print(f"\n→ retrieve: {sum((s.end_time - s.start_time) / 1e6 for s in retrieve):,.0f} ms of the request. Not the model.")
        paper_store.SEARCH_DELAY_S = SEARCH_DELAY_S = 0.0          # back to normal
    ''')

    # -- Section 7 -------------------------------------------------------------------------
    nb.section(7, "Evaluate it", "A golden set, scorers, a baseline and a gate: unit testing, for prompts.")
    nb.md("### 7.1 The golden set\n\nEight questions about TinyCoder with known right answers: six the paper answers "
          "(with the section the answer lives in) and two it can't, where the right answer is *the paper doesn't "
          "cover that*.")
    nb.anchor("7.1").code(r'''
        GOLDEN = golden_set("notebook")
        CHUNKS = chunk_index(papers.get(PAPER_ID))          # chunk id -> section, page, text
    ''', carry=True)
    nb.code(r'''
        # 7.1
        for q in GOLDEN:
            expected = ", ".join(q["expected_sections"]) if q["answerable"] else "refuse: the paper doesn't cover it"
            print(f"{q['id']}  {q['question']:60} → {expected}")
    ''')
    nb.md("### 7.2 Scorers and the LLM judge\n\nRules first, checked exactly by a program. Then one thing rules can't "
          "check: is every claim supported by the sources? A second model judges that, with its own versioned prompt. "
          "Free tier and 429s in 7.3? Set `EVAL_PAUSE_S = 4` here and run 7.2 and 7.3 again.")
    nb.anchor("7.2").code(consts(metrics, "JUDGE_VERSION", "REFUSAL", "HYPE", "REFUSED") + "\n\n\n"
                          + defs(metrics, "plain", "syllables", "reading_ease", "is_refusal", "score", "judge_messages",
                                 "parse_verdict", "judge") + r'''


JUDGE_V1 = load_prompt("judge", "v1")
EVAL_PAUSE_S = 0          # ← 4 on a free tier that allows ~20 requests per minute


def complete(messages, max_tokens):
    """The judge's model call: through the same gateway, capped at max_tokens."""
    return router.completion(model=ALIAS, messages=messages, max_tokens=max_tokens, temperature=0).choices[0].message.content''',
                          carry=True)
    nb.code(r'''
        # 7.2
        print(JUDGE_V1)
        for text in ("TinyCoder is small. It runs fast. It only knows Python.",
                     "TinyCoder uses sliding-window attention in every layer, with a 1,024-token window: each token "
                     "attends only to the 1,024 tokens before it."):
            print(f"\nreadability {reading_ease(text):5.1f}  {text[:90]}")
    ''')
    nb.md("### 7.3 Run it and save a baseline\n\nAll eight questions through the same agent, a tick or a cross each, then "
          "the averages go to `baseline.json` with the prompt version, the judge version and the model. A baseline "
          "isn't a target of 100%; valid citations is the exception: always 1.0.")
    nb.anchor("7.3").code(consts(runner, "NOTEBOOK_METRICS") + "\n\n\n" + defs(runner, "average", "run_eval"), carry=True)
    nb.code(r'''
        # 7.3 (about 22 model calls)
        eval_v1 = build_agent("v1", router=router, extra_middleware=[budget_guard])
        metrics_v1, rows_v1 = run_eval("v1", lambda q: ask_paper(PAPER_ID, q, "eval-bot", eval_v1), complete, CHUNKS, GOLDEN)
        BASELINE = baseline_record(metrics_v1, "v1", os.environ["LLM_MODEL"].split(",")[0])
        Path("baseline.json").write_text(json.dumps(BASELINE, indent=2))
        print("\nbaseline.json:", json.dumps(BASELINE, indent=1))
    ''')
    nb.md("""
        ### 7.4 Your turn: the regression gate. ✏️ TODO 3 (two lines)

        Answers wobble a little between runs, so a small drop is allowed: 0.10 for most metrics, 10 points for
        readability, **zero** for valid citations. **TODO 3:** if the current value is more than the allowed amount
        below the baseline, add a message to `failures`.
    """)
    answer3 = 'if current[metric] < base - allowed:\n            failures.append(f"{metric}: {base} -> {current[metric]}")'
    sol3 = consts(gate, "TOLERANCE") + "\n\n\n" + defs(gate, "regression_gate")
    nb.anchor("7.4").todo(stub=stub(sol3, answer3, "# ✏️ TODO 3 (two lines): if current[metric] is more than `allowed` "
                                                   "below `base`,\n#   append f\"{metric}: {base} -> {current[metric]}\" "
                                                   "to failures\npass"),
                          solution=sol3)
    nb.md("### 7.4a Test: offline\n\nSmall wobbles pass; real drops are caught; a fake citation is never allowed.")
    nb.todo_test(r'''
        # 7.4a test
        base = {"valid_citations": 1.0, "cited": 1.0, "right_section": 0.83, "readability": 48.0}
        wobble = {"valid_citations": 1.0, "cited": 0.95, "right_section": 0.75, "readability": 40.0}
        drop = {"valid_citations": 0.99, "cited": 0.0, "right_section": 0.83, "readability": 30.0}
        assert regression_gate(wobble, base) == [], f"TODO 3: wobbles inside the tolerance should pass, got {regression_gate(wobble, base)}"
        caught = [f.split(":")[0] for f in regression_gate(drop, base)]
        print("caught:", regression_gate(drop, base))
        assert caught == ["valid_citations", "cited", "readability"], f"TODO 3: expected valid_citations, cited and readability, got {caught}"
        print("✅ TODO 3 passed: wobbles pass, real drops and any fake citation are caught")
    ''')
    nb.rescue(sol3, "Two lines, inside the loop:\n\n```\nif current[metric] < base - allowed:\n"
                    "    failures.append(f\"{metric}: {base} -> {current[metric]}\")\n```")
    nb.md("""
        ### 7.5 A harmless prompt change

        A teammate: *"the answers look cluttered with those [p2-c1] markers, can we make it friendlier?"* Prompt v2 says:
        be friendly and conversational, no brackets or reference markers. Everything else is the same. Run it against
        the baseline.
    """)
    nb.anchor("7.5").code(r'''
        # 7.5 (about 22 model calls)
        print("v2's rule 3:", [line for line in load_prompt("agent", "v2").splitlines() if line.startswith("3.")][0], "\n")
        eval_v2 = build_agent("v2", router=router, extra_middleware=[budget_guard])
        metrics_v2, rows_v2 = run_eval("v2", lambda q: ask_paper(PAPER_ID, q, "eval-bot", eval_v2), complete, CHUNKS, GOLDEN)
        baseline = json.loads(Path("baseline.json").read_text())
        print()
        compare(baseline["metrics"], metrics_v2, ("v1", "v2"))
        failures = regression_gate(metrics_v2, baseline["metrics"])
        if failures:
            print("\n❌ BLOCKED: v2 can't ship. Worse than the baseline:", " · ".join(failures))
        else:
            print("\n✅ v2 passes the gate: on this model, the answers kept their quality (or TODO 3 isn't solved: run 7.4a)")
    ''')

    # -- Closing ---------------------------------------------------------------------------
    nb.md("---\n# Go live\n\n### C.1 The go-live checklist\n\nEach item, and the cell where you did it today. Two say "
          "*extension*: the first things to add next.")
    nb.code(r'''
        # C.1
        CHECKLIST = {
            "Serve": [("Auth on every endpoint", "4.2"), ("Validate uploads by bytes and size", "4.3"),
                      ("Hash: never process a paper twice", "4.4"), ("Stream long answers", "4.6"),
                      ("Clear errors, not stack traces", "4.2, 4.3a")],
            "Control": [("One gateway, one retry layer", "5.1"), ("Fallback model", "5.2"),
                        ("Budget checked before every call", "5.3, 5.4"), ("max_tokens on every call", "4.0, 5.4"),
                        ("Per-user rate limits with Retry-After", "5.5"), ("Spend visible per user", "5.6")],
            "Observe": [("One JSON log line per request", "6.1"), ("Request id in every response", "4.5, 6.1"),
                        ("Traces across every step", "6.2"), ("Tokens and cost on every LLM span", "6.2"),
                        ("Alerts: spend, errors, latency", "extension")],
            "Evaluate": [("Golden set, including refusals", "7.1"), ("Rules first, then a judge", "7.2"),
                         ("Version prompts and judges", "7.2, 7.5"), ("Baseline + gate before every change", "7.3-7.5"),
                         ("Weekly eval on sampled real traffic", "extension")],
            "Still on from Days 1-3": [("Untrusted-text guard, citation check, step limit", "4.0")],
        }
        for group, items in CHECKLIST.items():
            print(f"\n{group}")
            for item, cell in items:
                print(f"  ☐ {item:48} {cell}")
    ''')
    nb.md(f"""
        ### C.2 Production cheatsheet

        **Serve**
        - Auth on every endpoint ({nb.link("4.2")})
        - Validate uploads by content and size ({nb.link("4.3")})
        - Hash documents; never process one twice ({nb.link("4.4")})
        - Stream long answers ({nb.link("4.6")})
        - Turn every known error into a clear status and message ({nb.link("4.2")})

        **Control**
        - One gateway for every LLM call ({nb.link("5.1")})
        - Retries, timeout, fallback in one layer ({nb.link("5.1")}, {nb.link("5.2")})
        - Per-user budget checked before every call ({nb.link("5.3")})
        - `max_tokens` on every call (limits overshoot) ({nb.link("5.4")})
        - Per-user rate limits with Retry-After ({nb.link("5.5")})
        - Spend visible per user ({nb.link("5.6")})

        **Observe**
        - One JSON log line per request, with a request id ({nb.link("6.1")})
        - Keep document text and personal data out of general logs ({nb.link("6.1")})
        - Traces across parse, retrieve, LLM and guards ({nb.link("6.2")})
        - Tokens and cost on every LLM span ({nb.link("6.2")})
        - Alerts on spend, errors and latency

        **Evaluate**
        - Golden set, including questions to refuse ({nb.link("7.1")})
        - Rules first, an LLM judge for faithfulness ({nb.link("7.2")})
        - Version prompts and judge prompts ({nb.link("7.2")})
        - Baseline + regression gate before every change ({nb.link("7.3")}, {nb.link("7.4")})
        - Re-run on sampled real traffic to catch drift
    """)
    nb.md("""
        ---
        ### Extensions (homework)

        **E.1 A small front end.** The next cell writes `ui.py`, a Streamlit page that uploads a PDF and asks your
        service questions. Run it on your own machine: `pip install streamlit`, start the service (`uvicorn` or the
        Docker repo's `make up`), then `streamlit run ui.py`.

        **E.2 Real traces in Jaeger.** Swap `InMemorySpanExporter()` for an OTLP exporter pointed at Jaeger, and look
        at the same tree in a browser.

        **E.3 The real LiteLLM proxy.** The workshop repo's Docker setup runs the proxy with Postgres and Redis: one
        virtual key per user with `max_budget`, `budget_duration` and `rpm_limit` (`make up`, then `make keys`).

        **E.4 Grow the golden set to 20.** Add questions from your own use of the service, including two it should
        refuse, and re-run the gate. The repo's `make eval` runs a 20-question set the way CI would.
    """)
    nb.code(r'''
        %%writefile ui.py
        # E.1: a Streamlit front end for the Day 4 service. Run it locally: pip install streamlit; streamlit run ui.py
        import os

        import httpx
        import streamlit as st

        API = os.environ.get("API_URL", "http://127.0.0.1:8765")
        st.title("Paper explainer")
        st.caption("TinyCoder is a FICTIONAL paper written for the workshop. Never upload confidential documents.")
        key = st.text_input("API key", value="key-alice", type="password")
        headers = {"Authorization": f"Bearer {key}"}
        upload = st.file_uploader("Upload a PDF", type=["pdf"])
        if upload is not None and "paper_id" not in st.session_state:
            r = httpx.post(f"{API}/papers", headers=headers, timeout=120,
                           files={"file": (upload.name, upload.getvalue(), "application/pdf")})
            if r.status_code == 200:
                st.session_state.paper_id = r.json()["paper_id"]
                st.success(f"{r.json()['status']}: {st.session_state.paper_id}")
            else:
                st.error(r.json().get("message", r.text))
        question = st.text_input("Your question", placeholder="How much faster is it than the baseline?")
        if question and "paper_id" in st.session_state:
            r = httpx.post(f"{API}/papers/{st.session_state.paper_id}/ask", headers=headers, timeout=120,
                           json={"question": question})
            if r.status_code == 200:
                body = r.json()
                st.write(body["answer"])
                st.caption(f"citations {body['citations']} · {body['llm_calls']} LLM calls · ${body['cost_usd']:.6f} · "
                           f"answered by {', '.join(body['answered_by'])} · request {r.headers['X-Request-ID']}")
            else:
                st.error(f"{r.status_code}: {r.json().get('message', r.text)}")
        usage = httpx.get(f"{API}/me/usage", headers=headers, timeout=30)
        if usage.status_code == 200:
            st.sidebar.json(usage.json())
    ''')
    return nb


# ===========================================================================
# Instructor demo: the production stack (gateway + postgres + redis + api)
# ===========================================================================
DEMO_SETUP = r'''
# Setup: talks to the Docker stack (`make up`): the LiteLLM proxy, Postgres, Redis and the api in proxy mode.
# Inside the compose network (make exec-notebooks) the services are gateway:4000 and api:8000; from the host,
# localhost:4000 and localhost:8200. Keys come from the environment and keys.local.json; none is printed in full.
import json, os, subprocess, sys, time
from pathlib import Path
import httpx

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents, Path("/workspace")] if (p / "gateway" / "litellm_config.yaml").exists())
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
if not os.environ.get("LITELLM_MASTER_KEY") and (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("LITELLM_MASTER_KEY="):
            os.environ["LITELLM_MASTER_KEY"] = line.split("=", 1)[1].split("#")[0].strip()

def reachable(url):
    try:
        return httpx.get(url, timeout=3, trust_env=False).status_code < 500
    except httpx.HTTPError:
        return False

GATEWAY = os.environ.get("GATEWAY_URL") or ("http://gateway:4000" if reachable("http://gateway:4000/health/liveliness") else "http://localhost:4000")
API = os.environ.get("API_URL") or ("http://api:8000" if reachable("http://api:8000/health") else "http://localhost:8200")
os.environ["GATEWAY_URL"] = GATEWAY
from paper_agent.fixtures.tinycoder_pdf import make_pdf
from paper_agent.service import keys
from paper_agent.service.keys import mask
MASTER = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"}
gw = httpx.Client(base_url=GATEWAY, timeout=60, trust_env=False)
api = httpx.Client(base_url=API, timeout=120, trust_env=False)
Q_SPEED = "How much faster is TinyCoder than the 7B baseline?"
print("gateway:", GATEWAY, "| api:", API, "| master key:", mask(os.environ["LITELLM_MASTER_KEY"]))
'''


def demo() -> Notebook:
    nb = Notebook("day4-demo")
    nb.md("""
        # Day 4 Instructor Demo: the production version (optional)

        **Run by the instructor only, locally, against the Docker stack** (`make up`: gateway, postgres, redis, api, all
        upstream of mock-llm by default, so it rehearses with no provider key). The guide's talk sections are slides
        only; this notebook shows what participants did in library mode, done by the LiteLLM **proxy**: virtual keys,
        budgets and rate limits enforced on every call, shared by every app. Repeatable: run it top to bottom as often as
        you like (keys are recreated by alias). TinyCoder is fictional.
    """)
    nb.code(DEMO_SETUP)
    nb.md("---\n## D0 · Before class: is everything up?")
    nb.code(r'''
        # D0
        ready = gw.get("/health/readiness", headers=MASTER).json()
        cache = gw.get("/cache/ping", headers=MASTER).json()
        print("gateway  :", gw.get("/health/liveliness").json(), "|", ready.get("status"))
        print("postgres :", ready.get("db"), "(virtual keys, budgets, spend)")
        print("redis    :", cache.get("cache_type"), "ping", cache.get("ping_response"), "(rate-limit counters, response cache)")
        print("api      :", api.get("/health").json())
        print("aliases  :", sorted(m["id"] for m in gw.get("/v1/models", headers=MASTER).json()["data"]))
    ''')
    nb.md("---\n## D5.1 · The same gateway, as config · *slide: The gateway: Day 1, now as config*\n\nLeft: the Router "
          "participants print in 5.1. Right: `gateway/litellm_config.yaml`, which the proxy runs.")
    nb.code(r'''
        # D5.1
        import yaml
        cfg = yaml.safe_load((ROOT / "gateway" / "litellm_config.yaml").read_text())
        os.environ.setdefault("LLM_BASE_URL", "http://mock-llm:8000/v1"); os.environ.setdefault("LLM_API_KEY", "sk-mock-local")
        os.environ.setdefault("LLM_MODEL", "mock-llm"); os.environ.setdefault("LLM_FALLBACK_MODEL", "mock-llm-fallback")
        from paper_agent.service.gateway import make_router
        router = make_router()
        rs = cfg["router_settings"]
        rows = [("alias -> model", ", ".join(f"{d['model_name']}->{d['model_info']['id']}" for d in router.model_list if d["model_info"]["id"] != "embed"),
                 ", ".join(f"{m['model_name']}->{m['model_info']['id']}" for m in cfg["model_list"] if m["model_info"]["id"] not in ("embed", "broken-primary"))),
                ("fallbacks", json.dumps(router.fallbacks), json.dumps([f for f in rs["fallbacks"] if "paper-explainer" in f])),
                ("num_retries", router.num_retries, rs["num_retries"]),
                ("timeout (s)", router.timeout, rs["timeout"]),
                ("max_tokens", "400 (make_chat_model)", cfg["model_list"][0]["litellm_params"]["max_tokens"]),
                ("budgets", "our @before_model guard", "virtual keys (Postgres)"),
                ("rate limits", "in-process counter", "rpm_limit per key (Redis)")]
        print(f"{'':16} {'notebook: litellm.Router (library)':58} {'production: litellm_config.yaml (proxy)'}")
        for name, lib, proxy in rows:
            print(f"{name:16} {str(lib):58} {proxy}")
    ''')
    nb.md("---\n## D5.2 · Fallback through the proxy\n\n`paper-explainer-broken` points at a model the provider doesn't "
          "have; the proxy falls back. The response header says which deployment answered.")
    nb.code(r'''
        # D5.2
        r = gw.post("/v1/chat/completions", headers=MASTER, json={"model": "paper-explainer-broken", "max_tokens": 20,
                    "messages": [{"role": "user", "content": "Reply with one word: ready"}]})
        print("status:", r.status_code)
        for h in ("x-litellm-model-id", "x-litellm-model-group", "x-litellm-attempted-fallbacks", "x-litellm-attempted-retries"):
            print(f"{h:32} {r.headers.get(h)}")
    ''')
    nb.md("---\n## D5.6a · One virtual key per user · *slide: Limits and spend*\n\n`POST /key/generate`. Alice gets "
          "$1.00 a day and 10 requests a minute. Bob's budget is about 1.5 questions, measured from one real question.")
    nb.code(r'''
        # D5.6a
        ALICE_SPEC = {"user_id": "alice", "max_budget": 1.00, "budget_duration": "1d", "rpm_limit": 10}
        keys.delete_keys(["alice", "bob"])
        print("POST /key/generate", json.dumps({"key_alias": "alice", "models": keys.MODELS, **ALICE_SPEC}))
        ALICE_KEY = keys.generate_key("alice", **ALICE_SPEC)["key"]
        print("  → key", mask(ALICE_KEY))
        PAPER_ID = api.post("/papers", headers={"Authorization": f"Bearer {ALICE_KEY}"},
                            files={"file": ("tinycoder.pdf", make_pdf(), "application/pdf")}).json()["paper_id"]
        one = api.post(f"/papers/{PAPER_ID}/ask", headers={"Authorization": f"Bearer {ALICE_KEY}"}, json={"question": Q_SPEED}).json()
        BOB_SPEC = {"user_id": "bob", "max_budget": round(1.5 * one["cost_usd"], 6), "budget_duration": "1d", "rpm_limit": 10}
        print(f"\none question costs ${one['cost_usd']:.6f} → bob's budget ${BOB_SPEC['max_budget']:.6f}")
        print("POST /key/generate", json.dumps({"key_alias": "bob", "models": keys.MODELS, **BOB_SPEC}))
        BOB_KEY = keys.generate_key("bob", **BOB_SPEC)["key"]
        print("  → key", mask(BOB_KEY))
        _ = Path(keys.KEYS_FILE).write_text(json.dumps({**keys.load_keys(), "alice": ALICE_KEY, "bob": BOB_KEY}, indent=2))
    ''')
    nb.md("---\n## D5.6b · Bob runs out, through the api\n\nThe proxy refuses Bob's next model call; our api turns that "
          "into the same friendly 429 participants saw in 5.4. First the proxy's raw error, once.")
    nb.code(r'''
        # D5.6b
        BOB_H = {"Authorization": f"Bearer {BOB_KEY}"}
        for n in (1, 2, 3):
            r = api.post(f"/papers/{PAPER_ID}/ask", headers=BOB_H, json={"question": Q_SPEED})
            print(f"bob, question {n}: {r.status_code}", "" if r.status_code == 200 else r.json()["error"])
        raw = gw.post("/v1/chat/completions", headers=BOB_H, json={"model": "paper-explainer", "max_tokens": 5,
                      "messages": [{"role": "user", "content": "hi"}]})
        err = raw.json()["error"]
        print(f"\nthe proxy, raw : HTTP {raw.status_code} type={err['type']!r}\n                 {err['message'].split('(sk-')[0]}…")
        print("our api        :", r.status_code, json.dumps(r.json()))
    ''')
    nb.md("---\n## D5.6c · Rate limit through the proxy\n\nEvery model and embedding call counts against Alice's 10 "
          "requests per minute (one question is about three).")
    nb.code(r'''
        # D5.6c
        ALICE_H = {"Authorization": f"Bearer {ALICE_KEY}"}
        for n in range(1, 8):
            r = api.post(f"/papers/{PAPER_ID}/ask", headers=ALICE_H, json={"question": Q_SPEED})
            print(f"alice, question {n}: {r.status_code}", "" if r.status_code == 200 else
                  f"Retry-After: {r.headers.get('Retry-After')} · {r.json()['message']}")
            if r.status_code == 429:
                break
    ''')
    nb.md("---\n## D5.6d · Spend per key\n\n`GET /key/info` next to the api's `/me/usage`, which reads it in proxy mode. "
          "The proxy writes spend to Postgres in batches: the first read can lag a few seconds.")
    nb.code(r'''
        # D5.6d
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not all(keys.key_info(k)["spend"] for k in (ALICE_KEY, BOB_KEY)):
            time.sleep(3)
        for user, key in (("alice", ALICE_KEY), ("bob", BOB_KEY)):
            info = keys.key_info(key)
            me = api.get("/me/usage", headers={"Authorization": f"Bearer {key}"}).json()
            print(f"{user:6} /key/info spend ${info['spend']:.6f} of ${info['max_budget']} (resets {info.get('budget_reset_at', '?')[:10]})"
                  f"  |  /me/usage {json.dumps(me)}")
    ''')
    nb.md("---\n## D6 · The same spans in production\n\nThe api's code, in proxy mode, with the console exporter (the "
          "api service's default: `docker compose logs api`). Same span names as 6.2.")
    nb.code(r'''
        # D6
        from fastapi.testclient import TestClient
        from paper_agent.service import config
        from paper_agent.service.app import build_app
        from paper_agent.service.pipeline import build_agent
        from paper_agent.service.store import PaperStore, setup_retrieval
        from paper_agent.service.tracing import one_line, setup_tracing
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        config.GATEWAY_MODE, config.GATEWAY_URL = "proxy", GATEWAY
        setup_retrieval(None, "/tmp/d6")
        setup_tracing(ConsoleSpanExporter(formatter=one_line))
        CAROL_H = {"Authorization": f"Bearer {keys.recreate_keys({'carol': {'user_id': 'carol', 'max_budget': 1.0}})['carol']}"}
        with TestClient(build_app(build_agent(), PaperStore("/tmp/d6"))) as c:
            pid = c.post("/papers", headers=CAROL_H, files={"file": ("tinycoder.pdf", make_pdf(), "application/pdf")}).json()["paper_id"]
            print("---- one question ----")
            r = c.post(f"/papers/{pid}/ask", headers=CAROL_H, json={"question": "What attention mechanism does TinyCoder use?"})
        setup_tracing(ConsoleSpanExporter(formatter=lambda s: ""))       # quiet again
        print("status", r.status_code, "| answered_by", r.json().get("answered_by"))
    ''')
    nb.md("---\n## D7 · The gate in CI · *slide: The gate*\n\n`make eval`: the 20-question set through the api, metrics "
          "against `paper_agent/evals/baseline.json`, and an exit code a CI job can fail on. Then the v2 prompt.")
    nb.code(r'''
        # D7
        EVAL_KEY = keys.recreate_keys({"eval-bot": keys.DEMO_KEYS["eval-bot"]})["eval-bot"]
        for version in ("v1", "v2"):
            print(f"$ make eval PROMPT_VERSION={version}")
            run = subprocess.run([sys.executable, "-m", "paper_agent.evals.runner", "--api", API, "--gateway", GATEWAY,
                                  "--prompt-version", version], cwd=ROOT, capture_output=True, text=True,
                                 env={**os.environ, "EVAL_KEY": EVAL_KEY, "PYTHONPATH": str(ROOT)})
            print(run.stdout or run.stderr[-1500:])
            print(f"exit code: {run.returncode}  ({'the build passes' if run.returncode == 0 else 'CI fails the build'})\n")
    ''')
    nb.md("---\n## DX · After class: clean up\n\nDelete the demo keys, so the next rehearsal starts clean (`POST "
          "/key/delete` by alias).")
    nb.code(r'''
        # DX
        print("deleted:", keys.delete_keys(["alice", "bob", "carol", "eval-bot"]), "key(s)")
        r = httpx.get(f"{GATEWAY}/key/info", params={"key": BOB_KEY}, headers={"Authorization": f"Bearer {BOB_KEY}"}, trust_env=False)
        print("bob's old key now:", r.status_code, r.json()["error"]["type"] if r.status_code != 200 else "still works?!")
    ''')
    return nb
