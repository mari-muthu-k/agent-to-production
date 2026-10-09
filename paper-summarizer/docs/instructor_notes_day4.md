# Day 4 instructor notes: serve, control, observe, evaluate

For the instructor and TAs only (it contains the answer keys). It sits in `paper-summarizer/docs/`, which is not
part of the published course site.

Notebooks: `notebooks/day4/Day4_CodeAlong.ipynb` (code-along, cells 0.1–7.5, C.1–C.2, E.1) and
`notebooks/day4/Day4_Instructor_Demo.ipynb` (optional, D0–DX, the production stack). Both are generated from
`paper-summarizer/notebook_templates/day4.py` with `make notebooks DAYS=4`. Every number marked **mock** comes from
`make exec-notebooks` / `make eval` against mock-llm on 9 October 2026 (LiteLLM 1.104.2, LangChain 1.4.3,
langchain-litellm 0.11.0, FastAPI 0.142.4). **Nothing here has been run against a real provider yet**: the
"real model" lines are what to expect and what to check tonight, not measurements.

## Before the session

- **Push to `main` first.** Cell 4.0 downloads `requirements-day4.txt` and about 35 files of `paper_agent/`
  (Days 2–4: `rag/pdf_rag.py`, `agent/*`, `service/*`, `evals/*`, `prompts/agent_v1|v2|judge_v1.txt`,
  `fixtures/tinycoder_pdf.py`) from GitHub `main`. Until they are pushed, 4.0 fails with a 404.
- **The install (0.1).** `pip install -r requirements-day4.txt` into a bare `python:3.12-slim` container took
  **89 s**, and the first import of LiteLLM + LangChain + Chroma another **5 s** (same laptop and network as the
  Day 3 measurement of 55 s). Not yet timed in a fresh Colab runtime: Colab preinstalls FastAPI, httpx, pydantic
  and OpenTelemetry at other versions, so 0.1 upgrades them and then asks for **Runtime → Restart session, then
  4.0**. Expect most people to see that message; it is normal.
- **Secrets.** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL` as on Days 2–3; optional
  `LLM_FALLBACK_MODEL`, `PRICE_IN_PER_M`, `PRICE_OUT_PER_M` (empty = $0.15 / $0.60).
- **Gemini.** `make_router()` sends Gemini (`generativelanguage.googleapis.com`) through LiteLLM's native
  `gemini/` provider instead of the OpenAI-compatible endpoint, because that provider round-trips Gemini's
  tool-call thought signatures (the Day 3 bug). **Unverified**: run 4.1 and 7.3 with a Gemini key tonight.
- **Load.** One participant makes **~74 chat calls** (table below), 24 of them in 7.3 and 24 in 7.5, each pair
  inside a minute or two. Free tiers at ~20 requests per minute: `EVAL_PAUSE_S = 4` in 7.2.

## Cell-to-slide map

| Time | Step | Slide | Cells |
|---|---|---|---|
| 2:00–2:30 | 1.1–3.4 | Four days, one agent → Evaluate like you test | slides only (0.1 install running in the background) |
| 2:30 | 4.1 | The service | 4.0 setup/catch-up, 4.1 setup check ("✅ ready") |
| 2:33 | 4.2 | The service | 4.2 `current_user`, `error_response`, `build_app`; prints the four routes |
| 2:37 | 4.3 | TODO 1: never trust the file | 4.3 **TODO 1**, 4.3a test (415 / 413 / 401), 4.3b rescue |
| 2:42 | 4.4 | Upload, ask, stream | 4.4 `start_server` (port 8765), the same PDF under two names |
| 2:45 | 4.5 | Upload, ask, stream | 4.5 ask over HTTP: `X-Request-ID`, usage fields |
| 2:47 | 4.6 | Upload, ask, stream | 4.6 SSE with timestamps |
| 2:50 | 5.1 | The gateway: Day 1, now as config | 5.0 catch-up, 5.1 Router + mapping table |
| 2:55 | 5.2 | The gateway: Day 1, now as config | 5.2 forced fallback (`answered_by: ['fallback']`), primary restored |
| 2:58 | 5.3 | A budget on every call | 5.3 **TODO 2**, 5.3a test, 5.3b rescue |
| 3:02 | 5.4 | A budget on every call | 5.4 guard joins the agent; Bob 200, 200, 429; Alice 200; overshoot line |
| 3:07 | 5.5 | Limits and spend | 5.5 three per minute: 200 ×3, 429 ×2 with `Retry-After: 60` |
| 3:10 | 5.6 | Limits and spend | 5.6 `/me/usage` for both (Bob negative) + the proxy slide |
| 3:12 | 6.1 | One line per request | 6.0 catch-up, 6.1 log handler + the support-ticket lookup |
| 3:17 | 6.2 | Where did the time go? | 6.2 the span lines from 4.0's code, then the trace tree |
| 3:23 | 6.3 | Where did the time go? | 6.3 `SEARCH_DELAY_S = 1.5`: the long `retrieve` bar; reset |
| 3:27 | 7.1 | Golden set and scores | 7.0 catch-up, 7.1 the eight questions |
| 3:30 | 7.2 | Golden set and scores | 7.2 scorers, `JUDGE_V1`, `EVAL_PAUSE_S`, two readability examples |
| 3:34 | 7.3 | Golden set and scores | 7.3 run v1, ticks/crosses, `baseline.json` |
| 3:38 | 7.4 | The gate | 7.4 **TODO 3**, 7.4a test, 7.4b rescue |
| 3:42 | 7.5 | The gate | 7.5 run v2, comparison table, "❌ BLOCKED" |
| 3:47 | 8.1 | Go-live checklist | C.1 checklist (cells named), C.2 cheatsheet |
| 3:51+ | 8.2–8.3 | What you built | E.1 Streamlit `ui.py` (if running early, run it locally) |

## Answer key

| TODO | Answer |
|---|---|
| 4.3 TODO 1 | `if not data.startswith(b"%PDF-"): raise HTTPException(415, "Only PDF files are accepted.")` then `if len(data) > MAX_UPLOAD_MB * 1024 * 1024: raise HTTPException(413, f"Max upload size is {MAX_UPLOAD_MB} MB.")` |
| 5.3 TODO 2 | `if spent >= budget:` / `    raise BudgetExceededError(user_id, spent, budget)` |
| 7.4 TODO 3 | `if current[metric] < base - allowed:` / `    failures.append(f"{metric}: {base} -> {current[metric]}")` |

All three tests are offline: 4.3a uses FastAPI's `TestClient` (no server, no model call), 5.3a uses
`FakeToolModel` and `get_section`, 7.4a uses plain dicts. Unsolved, they fail with:

- 4.3a: `AssertionError: TODO 1: check the first bytes: data.startswith(b'%PDF-'), else HTTPException(415, ...)`
  (the text file reaches the parser and comes back as a 500)
- 5.3a: `AssertionError: TODO 2: spent $0.001 of a $0.001 budget should raise BudgetExceededError`
- 7.4a: `AssertionError: TODO 3: expected valid_citations, cited and readability, got []`

With TODO 3 unsolved, 7.5 prints "✅ v2 passes the gate … (or TODO 3 isn't solved: run 7.4a)" (guide, Common
problems).

## What each failure cell prints

| Cell | mock (measured) | real model (expected; check tonight) |
|---|---|---|
| 4.3a | `415 unsupported_type`, `413 too_large`, `401 unauthorized` | identical: no model involved |
| 4.4 | both names → `c03739855bbd  already processed` (4.0 processed it) | identical: the PDF's bytes are fixed, so the id is too |
| 5.2 | `answered_by: ['fallback']`; the mock saw a 404 then the fallback for each of the 2 agent steps | 404 or 400 from the provider → `['fallback']`; see the next section |
| 5.4 | Bob's budget $0.000498 (1.5 × $0.000332); 200 (spent $0.000332), 200 ($0.000664), 429 "You've used your budget for today ($0.000664 of $0.000498). It resets at midnight UTC; ask your admin if you need more."; Alice 200; over by $0.000166 | same shape. If the first call of a question costs more than half the question (a model that writes long tool arguments), Bob's 429 can come in the middle of question 2 instead of at question 3; say "the check runs before every call" |
| 5.5 | 200 ×3, then 429 ×2 `Retry-After: 60s · Too many requests (3 per minute). Try again in 60 seconds.` | identical: no model involved |
| 6.3 | `retrieve` ≈ 1,513 ms of a 1,917 ms request; "→ retrieve: 1,513 ms of the request. Not the model." | the LLM bars are longer (hundreds of ms to seconds), `retrieve` still ≈ 1,500 ms + embedding latency |
| 7.5 | v1 cited 1.0 / right_section 0.83 → v2 0.0 / 0.0: "❌ BLOCKED: v2 can't ship. Worse than the baseline: cited: 1.0 -> 0.0 · right_section: 0.83 -> 0.0" | see below |

The mock follows the system prompt's rules, not magic words: v1 says "cite the chunk id … in square brackets",
v2 says "no brackets or reference markers", and the refusal sentence is read from the prompt itself.

## 5.2 on a real gateway

`make_router(primary="this-model-does-not-exist")` sends that name to your provider. OpenAI-compatible providers
(OpenRouter, Gemini's endpoint) normally answer **404 or 400** for an unknown model, and LiteLLM falls back on
both, without retrying the primary first (on the mock: exactly 2 calls per model call). **Some gateways remap
unknown names to a default model** and answer 200: then `answered_by` stays `['primary']`. Narrate it as the guide
says: "your gateway quietly served a default model; with a real outage, the fallback config above is what would
answer". Check tonight with your key: run 5.2 alone after 4.0. The proxy demo (D5.2) shows the same thing with
`x-litellm-model-id: fallback`.

## 7.5 on the real workshop model

Not measured (no provider key in this environment). On the mock v2 is blocked on **cited** and **right_section**
(both 1.0/0.83 → 0.0); readability is the same, because the mock writes the same sentences. On a real model,
expect cited and right_section to drop, and readability to go up a few points (the slide's 48 → 55). If the model
keeps citing anyway, v2 passes, which the guide treats as a valid result. Write the outcome here tonight.

## Cost per participant

Measured from mock token counts (they are close to a real BPE for English), at the workshop prices
$0.15 / $0.60 per 1M tokens:

| Cells | Chat calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| 4.1, 4.5, 6.1, 6.2, 6.3 (one question each) | 10 | 8,169 | 735 | $0.0017 |
| 4.6 (explain: 3 tools, 4 calls) | 4 | 4,381 | 176 | $0.0008 |
| 5.2 (fallback: 404 + fallback ×2, then the check) | 6 | 3,290 | 306 | $0.0007 |
| 5.4 (Bob ×2, Alice ×1; Bob's third never reaches the model) | 6 | 4,935 | 459 | $0.0010 |
| **7.3** (8 questions × 2 agent calls + 8 judge calls) | **24** | **18,742** | **1,054** | **$0.0034** |
| **7.5** (the same with v2) | **24** | **18,316** | **964** | **$0.0033** |
| **Total** | **74** | **57,833** | **3,694** | **$0.011** |

Plus 10 embedding requests (682 tokens). The guide estimates ~65 calls and ~$0.02 (1,500 input tokens per call);
measured is **74 calls and ~$0.011** (~780 input tokens per call: the three-page paper is small). A real model
that takes 3 steps instead of 2 or writes up to `max_tokens = 400` per answer could double that: budget
**$0.02–0.03 per person, $4–6 for 200**.

## The production stack (instructor demo, `make up`)

| Service | Image | Role |
|---|---|---|
| gateway | `ghcr.io/berriai/litellm:v1.104.2` (1.72 GB) | the proxy: aliases, retries, fallback, keys, budgets, limits |
| postgres | `postgres:16-alpine` (389 MB) | keys, budgets, spend |
| redis | `redis:7.4-alpine` (59 MB) | rate-limit counters, response cache (opt-in) |
| api | `paper-summarizer-runtime` (1.26 GB, was 464 MB) | the service, `GATEWAY_MODE=proxy` |

Cold `docker compose up` with images cached and empty volumes: **25 s** to all healthy. Run the demo
notebook with `make exec-notebooks NB=day4/Day4_Instructor_Demo` (executed copy in `build/executed/`), or open
it in VS Code attached to the `tests` container; it also works from the host against `localhost:4000/8200` with
a Python that has the pins installed.

Measured on proxy 1.104.2:

- **Over budget:** HTTP **422**, `error.type = "budget_exceeded"`, message
  `Budget has been exceeded! Key=bob (sk-...) Current cost: 0.000663, Max budget: 0.000498`. No `Retry-After`.
  The api maps it to the same 429 `budget_exceeded` body as 5.4.
- **Rate limited:** HTTP 429, `error.type = "throttling_error"`,
  `Rate limit exceeded for api_key: … Limit type: requests. Current limit: 2, Remaining: 0 …`, header
  `Retry-After: 60`. The api maps it to 429 `rate_limited` and passes `Retry-After` on.
- **Who answered:** response header `x-litellm-model-id` (`primary` / `fallback`: the `model_info.id` in
  `litellm_config.yaml`), plus `x-litellm-attempted-fallbacks: 1` on a fallback.
- **Spend is enforced at once but written to Postgres in batches:** `/key/info` showed `spend: 0.0` a few
  seconds after a call the budget check had already counted. D5.6d polls for up to 90 s.
- **Without a database** (`DATABASE_URL` unset), `max_end_user_budget` is still **not enforced** on 1.104.2:
  four calls of $2.55e-6 each against a $1e-7 end-user budget all returned 200. Library mode in Colab stays.
- The proxy refuses to start only with an unset, empty or publicly known master key (`sk-1234`);
  `.env.example` has a generated placeholder. `LITELLM_SALT_KEY` is set too (Postgres stores encrypted secrets).
- The Redis response cache is configured with `mode: default_off` (opt-in per request): a cached reply costs $0,
  which would hide Bob's budget and make `make eval` measure the cache.

## Concepts only (slide 2.3, anticipated questions)

- **vLLM and continuous batching.** A self-hosted inference server keeps the GPU busy by adding new requests to
  the batch at every decoding step instead of waiting for a whole batch to finish; PagedAttention stores the KV
  cache in pages so many sequences fit in memory. It pays off with high, steady volume, data-residency rules or a
  fine-tuned open model; you then own GPUs, scaling and upgrades. LiteLLM can sit in front of it like any provider.
- **Provider Batch APIs.** Upload a file of requests, collect the results within hours (typically within 24 h),
  at a large discount (often about half price). Good for offline work such as re-running the eval nightly or
  summarizing a library overnight; useless for anything a person is waiting for.

## Known gaps

- The reference code-along notebook (`reference/Day4_CodeAlong.ipynb`) was not available; cells follow the guide
  and the slides. Chunk ids are Day 2–3's `p3-c1` style, not the slides' illustrative `c4`.
- No real-provider run yet: 4.1, 5.2, 7.3 and 7.5 need a rehearsal with the workshop key tonight.
- The executed backup (`notebooks/day4/executed/Day4_CodeAlong_solved.ipynb`) is written by
  `make exec-notebooks` from **mock** outputs and is gitignored. For the session, keep the copy you execute in
  Colab with a participant key tonight (guide: "Tonight").
