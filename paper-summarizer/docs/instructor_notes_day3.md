# Day 3 instructor notes: agents, tools, safety and caching

For the instructor and TAs only (it contains the answer keys). It sits in `paper-summarizer/docs/`, which is
not part of the published course site.

Notebooks: `notebooks/day3/Practice.ipynb` (code-along, cells 3.0–6.5) and `notebooks/day3/Demo.ipynb`
(D2.2, D2.6). Both are generated from `paper-summarizer/notebook_templates/day3.py` with `make notebooks DAYS=3`.
Every number marked **mock** below comes from `make exec-notebooks` against mock-llm (LangChain 1.4.3,
langchain-core 1.6.7, LangGraph 1.2.14, langchain-openai 1.6.7, embeddings `mock-embed`).

## Before the session

- **Tool calling.** The workshop model must support tools (`make check-gateway` reports it). In Colab, run 3.1
  with the pinned model before the day.
- **Daily request caps.** A participant makes **about 70 chat calls** in the code-along (table below). OpenRouter's
  free tier allows **50 requests per day** per account without purchased credits, so free OpenRouter accounts
  without credits run out during Section 5. Use a key with credits, Gemini's free tier, or the gateway.
- **The install.** `pip install -r requirements-day3.txt` into a bare `python:3.12-slim` container took **55 s**
  (118 packages, from a laptop on a corporate network). Colab already has some of these preinstalled, so expect
  about the same or less; it has not been timed in a fresh Colab runtime yet. If LangChain was already loaded in
  the runtime, 3.0 says so and asks for **Runtime → Restart session, then 3.0**. Outside Colab, 3.0 never
  installs anything: it stops and says which versions differ.
- **USE_PRECOMPUTED.** `notebooks/day3/tinycoder_embeddings_day3.mock-embed.json` exists for `mock-embed` only.
  For the real embedding model, run once with your key:
  `EMBED_MODEL=<model> python tools/make_precomputed_embeddings.py --day 3`. Even then it rebuilds the index and
  the fixed questions only; agent searches use queries the model writes, so they need the embedding API.
- **Files Colab downloads** (3.0, from GitHub `main`): `requirements-day3.txt`, `llm_client.py`, a small slice of
  `paper_agent/` (Day 2's `pdf_rag.py`, imported as `day2`; `agent/policy.py`, `agent/fakes.py`,
  `agent/trace.py`), the three Day 3 PDFs and the precomputed vectors. Push before the session.

## Cell-to-slide map

| Time | Step | Slide | Cells |
|---|---|---|---|
| 2:10 | 2.2 | How tool calling works | Demo **D2.2** (raw request and reply, plain `openai` SDK) |
| 2:26 | 2.6 | Four places to cache | Demo **D2.6** (a five-line exact cache: miss, hit, one-word miss) |
| 2:30 | 3.1 | Tools: the docstring is the interface | 3.0 setup/catch-up, 3.1 check, 3.2 tools and the docstring demo |
| 2:37 | 3.3 | The model only proposes | 3.3 `bind_tools`: empty content, two `tool_calls` |
| 2:41 | 3.4 | Your turn: describe the tool | 3.4 **TODO 1** + test + rescue, 3.5 tool error as message |
| 2:50 | 4.1 | create_agent and the trace | 4.0 catch-up, 4.1 agent + streamed trace, 4.2 trace table |
| 2:57 | 4.3 | Memory in code | 4.3 threads, 4.4 store + `save_note` + profile |
| 3:05 | 4.5 | Failure: the runaway loop | 4.5 (`recursion_limit=25`, GraphRecursionError, cost line) |
| 3:10 | 5.1 | Guards are middleware | 5.0 catch-up, 5.1 hook map, 5.2 limits, 5.3 PII |
| 3:17 | 5.4 | Text aimed at the agent | 5.4 naive agent, 5.5 detector + **TODO 2** + test + rescue + rerun |
| 3:27 | 5.6 | Approval gate and citation check | 5.6 interrupt → reject → resume, 5.7 citation check, 5.8 full stack |
| 3:36 | 6.1 | Exact cache and the key | 6.0 catch-up, 6.1 baseline, 6.2 exact, 6.3 tool cache + **TODO 3** + test + rescue |
| 3:45 | 6.4 | Semantic and prompt caching | 6.4 semantic (calibrated), 6.5 prompt caching |
| 3:52 | C.1 | Production cheat sheet | Cheatsheet cell (links to the cells), extensions, Day 4 preview |

## Answer key

| TODO | Answer |
|---|---|
| 3.4 TODO 1 | `"""Look up how THIS paper defines a technical term, such as pass@1 or sliding-window attention.` / `Use it whenever the reader asks what a term means.` / `Returns the defining sentence with its section and page, or 'not defined in this paper'."""` |
| 5.5 TODO 2 | `result.content = wrap_untrusted(result.content, source=request.tool_call["name"])` then `if flags: result.content += guard_note(flags)` |
| 6.3 TODO 3 | `key = make_key(paper_hash, EMBED_MODEL, k, normalize(query))` |

The three TODO tests run offline: 3.4 and 5.5 use `FakeToolModel` (`paper_agent/agent/fakes.py`), 6.3 a fake
search function. They still work if the gateway is down. The 3.4 fake picks `define_term` only when the
docstring has a sentence saying **when** to use it that mentions meanings, definitions or terms (the same rule
mock-llm uses). A docstring with only "what" and "returns" fails, which is the point of the exercise.

## What the failure cells print

### 4.5 the runaway loop

**Mock** (deterministic): 13 `search_paper` calls with a new query each time, then

```
💥 GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
💰 13 model call(s), 12,194 input + 220 output tokens = $0.0065 at Day 1 prices
```

**Real model** (not measured yet): many models retry three to eight times and then answer "I couldn't find
it". That still shows the cost of a loop, but it may end without the GraphRecursionError. Say so if your
rehearsal run ends early.

**5.2 with limits (mock):** six model calls, then the run ends with the middleware's own message,
`Model call limits exceeded: run limit (6/6)`, and `💰 6 model call(s), 5,313 input + 92 output tokens = $0.0028`.
`recursion_limit` stays at 100 there on purpose: every middleware hook is a graph step, so with the limits added
a limit of 25 is reached before the sixth model call (checked against LangGraph 1.2.14).

### 5.4 the naive agent on the injected PDF

**Mock** (default `obey` mode): `search_paper` → `get_section(discussion)` →
`save_note(note='This reader never wants limitations in summaries.')` → `get_section(limitations)` → answer
*"TinyCoder is better than all large models at Python code completion [p4-c1]. It is small and fast [p4-c1]."*,
then `summary keeps the caveats: False | says 'better than all': True | notes saved: [...]` and `➡️ post in the chat: obeyed`.
The cell then deletes reader-3's poisoned note so later cells start clean.

**Real model:** varies between runs. Ask the room for the split, as the guide says. Rehearse twice and note both.

**5.5 rerun and 5.8 (mock):** both tool results show `+ 🛡 GUARD NOTE`, the summary keeps its caveats and ends
with *"(The document contains hidden instructions aimed at AI tools; I treated them as data and ignored them.)"*,
no note is saved: `resisted`. With TODO 2 unsolved, the rerun still obeys (the guard does nothing yet).

### 5.6 approval gate (mock)

The run pauses at `save_note`; the cell prints the interrupt payload (`action_requests` with the note,
`review_configs` with `allowed_decisions: approve, edit, reject, respond`), resumes on the same config with
`Command(resume={"decisions": [{"type": "reject", "message": ...}]})`, and ends with `notes saved for reader-3: none`.
With a real model that doesn't try `save_note`, the cell says so instead of failing.

### 5.7 citation check (mock, and any model)

A demo middleware (`cite_chunk_99`) appends `See also [c99].` to the final answer, so this works on every model.
The reader sees *"I couldn't verify the citations in this answer: it cites ['c99'], which were not retrieved in
this conversation turn. ..."*. (mock-llm also has a `force_citation` switch for tests.)

### 6.3 wrong-paper cache hit (mock and real: it's our cache, not the model)

```
tinycoder  MISS ['p4-c1', 'p1-c3', 'p5-c1', 'p5-c3']  '4 Results TinyCoder solves 41% of functions on the first att'
quickembed HIT  ['p4-c1', 'p1-c3', 'p5-c1', 'p5-c3']  '4 Results TinyCoder solves 41% of functions on the first att'
```

With TODO 3 solved, QuickEmbed gets its own chunks, and the repeat on QuickEmbed is a HIT.

### 6.4 semantic cache: calibrated thresholds

The cell measures two similarities with the participant's own embedding model and sets both thresholds from them:
loose = the lower score − 0.05; recommended = halfway between the reworded question and the near-miss.

| Embedding model | "What pass@1 does TinyCoder get?" | "What's TinyCoder's pass@10?" | loose | recommended |
|---|---|---|---|---|
| `mock-embed` (measured) | 0.713 | 0.444 | 0.39: both hit (pass@10 gets the pass@1 answer) | 0.58: reworded hits, pass@10 misses |
| the workshop's real model | **not measured yet** | | | |

Run 6.4 once with the real embedding model before the day and fill in that row. If the near-miss scores at least
as high as the rewording (it can, because numbers barely move an embedding), the cell prints that no threshold
separates them and that a semantic cache isn't safe for numeric questions on that model. That outcome is also a
valid lesson. Don't change the code live.

### 6.5 prompt caching

Mock: `What is the main result? input 2,285 tokens; cached: 0`, then `What are the limitations? input 2,284 tokens;
cached: 2,176`. mock-llm reports cached tokens when a request repeats ≥ 1,024 tokens of the previous one (blocks of
128). Real providers vary. `cached_input_tokens()` returns `None` when nothing is reported, and the cell prints
"your provider didn't report cached tokens". Note your gateway's number during rehearsal.

## Common problems, reproduced against the pinned versions

| Guide row | Reproduced? | What actually happens (LangChain 1.4.3 / LangGraph 1.2.14) |
|---|---|---|
| ImportError: create_agent or a middleware name | Checked by version | `ToolErrorMiddleware` first ships in **langchain 1.3.14**; older 1.x raises ImportError on it. 3.0 now detects version drift and asks for a restart. |
| 400 mentioning tools | Partly | The mock can only send a generic 400 (`X-Mock-Status: 400`); ChatOpenAI raises `BadRequestError` with the provider's message. |
| Agent answers without calling a tool | No | The mock always uses a tool when tools are offered. |
| GraphRecursionError in 4.5 | Yes | Expected, caught by `stream_trace` and printed with the cost line. |
| Interrupt never appears in 5.6 | **Differs** | Without a checkpointer the interrupt **does** appear (`result["__interrupt__"]`); resuming then raises `RuntimeError: Cannot use Command(resume=...) without checkpointer`. `build_agent()` always passes one. |
| Resume in 5.6 does nothing | **Differs** | `Command(resume=...)` on another thread_id starts a fresh run on an empty thread (on the mock it fails with a 400 from the embedding call); the paused run stays paused on its own thread. |
| Citation check blocks a good answer | Yes | `[limitations]` (a section name) fails the check; `[REDACTED_EMAIL]` is ignored (tested). |
| Cache never hits in 6.2 | Yes | Same prompt at temperature 0 then 0.2: a miss (one request sent). |
| Cached tokens 0 in 6.5 | Yes | 0 on the first call; `None` when the field is missing (tested). |
| 429s in 4.1 or 5.8 | Yes | ChatOpenAI retries twice (3 attempts served 429), then the run raises `RateLimitError`. |
| NameError anywhere | Yes | Every catch-up cell (4.0, 5.0, 6.0) runs clean from a fresh kernel in `make exec-notebooks`. |

## Cost per participant (mock, Day 1 prices: $0.50 / 1M input, $2.00 / 1M output)

One solved run of the code-along, catch-up and rescue cells not run:

| Cell | Calls | Input | Output | Cost |
|---|---:|---:|---:|---:|
| 3.1–3.3 | 4 | 913 | 67 | $0.0006 |
| **4.1 agent** | 4 | 6,096 | 166 | **$0.0034** |
| 4.3 memory | 5 | 6,216 | 292 | $0.0037 |
| 4.4 profile | 6 | 6,342 | 242 | $0.0037 |
| 4.5 runaway | 13 | 12,194 | 220 | $0.0065 |
| 5.2–5.7 | 26 | 35,138 | 733 | $0.0190 |
| **5.8 full stack** | 4 | 6,833 | 195 | **$0.0038** |
| 6.1–6.5 | 8 | 9,524 | 303 | $0.0054 |
| **Total** | **70** | **83,256** | **2,218** | **$0.046** |

Plus 13 embedding requests (2,726 tokens). Real models write longer answers than the mock (a few hundred output
tokens per final answer instead of 100–200), so budget about **$0.05–0.07 per participant** at these prices, about
$10–14 for 200 people, before reruns. The biggest burst is 4.5 (13 calls in a few seconds per participant), not
4.1 or 5.8.
