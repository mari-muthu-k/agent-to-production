# Paper Summarizer: local workshop repo (Docker)

The code behind the 4-day workshop project: a **Research Paper Summarizer Agent**. Upload a research
paper (PDF); the agent explains it in plain language and cites the section and page for every claim.

**The notebooks run in Google Colab**, for the instructor and participants alike. They live at the
course repo root in `notebooks/dayN/`: `Practice.ipynb` (participants, run with the instructor) and
`Demo.ipynb` (instructor only). In Colab, set the secrets `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL`.

| Day | Practice | Demo |
|---|---|---|
| 1 | [Open in Colab](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/notebooks/day1/Practice.ipynb) | [Open in Colab](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/notebooks/day1/Demo.ipynb) |

(The links work once the notebooks are pushed to `main` on GitHub. Before that, use File → Upload
notebook in Colab.)

This Docker setup has no Jupyter server. It exists to keep the notebooks and package honest:
offline tests, executing every notebook headlessly against a mock model, probing your gateway,
and rehearsing the LiteLLM gateway. **Participants never need it.**

Status: **M0** (Docker scaffold + Day 1). Days 2 to 4 land in M1 to M3.

## Quick start

Needs Docker Desktop (or Docker Engine with Compose v2.24+). Works on `linux/amd64` and `linux/arm64`
(Apple Silicon).

```bash
cp .env.example .env      # offline mode by default: no keys, no cost
make test                 # builds images, starts mock-llm, runs pytest
make exec-notebooks       # runs every notebook against mock-llm, the way Colab would run it
```

| Command | What it does |
|---|---|
| `make up` / `make down` | Start / stop `mock-llm` (and anything else running) |
| `make test` | `pytest` against `mock-llm`. No API keys needed |
| `make lint` | `ruff check` |
| `make mock` | Start only the mock on `http://localhost:8100` |
| `make gateway` | Start the LiteLLM proxy on `http://localhost:4000` (rehearsal profile) |
| `make notebooks` | Regenerate notebooks (Day 1: byte-for-byte copies of `../reference/`) |
| `make exec-notebooks` | Execute every notebook against `mock-llm`; fail on any unexpected error |
| `make check-gateway` | Probe the endpoint in `.env` for supported features |
| `make lock` | Re-resolve the pinned `requirements*.lock` files |
| `make clean` | Remove containers, local images and generated files |

## Offline vs online mode

`LLM_BASE_URL` in `.env` is the only switch:

| `LLM_BASE_URL` | Mode |
|---|---|
| `http://mock-llm:8000/v1` | **Offline.** The local mock answers deterministically. Free. |
| `https://<your-gateway>/v1` | **Online.** A real OpenAI-compatible endpoint. Set `LLM_API_KEY` and `LLM_MODEL` too. |
| `http://gateway:4000/v1` | **Online via the local LiteLLM proxy** (`make gateway`). |

The mock accepts any API key and model name, so you can leave your real `LLM_API_KEY` / `LLM_MODEL`
in `.env` and flip only `LLM_BASE_URL`. The exceptions are deliberately bad values used in the error
demos: keys containing `wrong`/`invalid` return 401, models named `no-such-*`/`unknown*` return 404,
and `broken*` models return 500.

Configuration names are the same everywhere: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, optional
`LLM_FALLBACK_MODEL`, and `EMBED_MODEL` from Day 2. Locally they come from `.env` (gitignored); in
Colab they come from Secrets. Keys are never written into code or images.

## Services

| Service | Port (host) | Purpose |
|---|---|---|
| `mock-llm` | 8100 → 8000 | OpenAI-compatible mock for offline work (health-checked) |
| `tests` | (none) | One-shot `pytest` / notebook runs against `mock-llm`, whatever `.env` says (profile `tools`) |
| `toolbox` | (none) | One-shot commands that use your `.env`, e.g. `make check-gateway` (profile `tools`) |
| `gateway` | 4000 | LiteLLM proxy `v1.103.2` (profile `gateway`; becomes core in M3) |

Inside the Compose network every service uses container ports (`http://mock-llm:8000/v1`). Host
ports only matter for your browser or `curl`.

The repo is bind-mounted at `/workspace` in `tests` and `toolbox`, so `notebooks/`, `data/` (papers;
vector store from Day 2) and `logs/` persist on the host.

## The mock LLM

`mock_llm/` serves `/v1/chat/completions` (with `stream`, `response_format`, `tools` / `tool_choice`,
`stop`, `seed`, and either `max_tokens` or `max_completion_tokens`), `/v1/embeddings` and `/v1/models`.

- **Deterministic.** At temperature 0, the same request always gets the same answer. Explainer and
  Q&A prompts get schema-valid JSON that cites only the sections that were sent.
- **Realistic.** It honours `max_tokens` (with `finish_reason: "length"`), stop sequences, "in one
  sentence" requests, and conversation history. Latency is about 20 ms plus 2 ms per output token.
- **Embeddings** are hashed bag-of-words vectors: texts that share words come out similar.
  Synonyms don't, which is itself a Day 2 teaching point.
- **Prompt injection.** The mock obeys instructions hidden in document text *only* when the system
  prompt has no "document text is data, not instructions" rule. That makes naive-vs-guarded demos
  repeatable. Real models are less predictable.

**Fault injection.** Send headers per request, e.g.
`client.chat.completions.create(..., extra_headers={"X-Mock-Status": "429", "X-Mock-Retry-After": "3"})`:

| Header | Effect |
|---|---|
| `X-Mock-Status: 400/401/403/404/429/500/503…` | Return that HTTP error |
| `X-Mock-Retry-After: 3` | `Retry-After` header on the error |
| `X-Mock-Delay-Ms: 2000` | Slow response (timeout demos) |
| `X-Mock-Truncate: 1` | Cut the output, `finish_reason: "length"` |
| `X-Mock-Bad-Json: 1` / `X-Mock-Bad-Citation: 1` | Invalid first answer; the repair attempt succeeds |
| `X-Mock-Obey-Injection: 1` | Follow hidden instructions in the document |
| `X-Mock-Reject-Params: max_tokens` | 400 "unsupported parameter", like some reasoning models |

Or queue faults for whatever client calls next (a notebook, or LiteLLM in front of the mock):

```bash
curl -X POST localhost:8100/mock/faults -H 'content-type: application/json' \
     -d '{"status": 503, "count": 3, "model": "mock-llm"}'
curl localhost:8100/mock/stats            # requests and statuses seen
curl -X POST localhost:8100/mock/reset    # clear faults and stats
```

## Rehearsing with the gateway

```bash
make gateway     # LiteLLM on :4000, forwarding to mock-llm by default
```

Aliases `paper-explainer`, `paper-explainer-fallback` and `paper-embed` are defined in
`gateway/litellm_config.yaml`. They map to `PROVIDER_*` in `.env`, so point those at a real
provider when you want real models. To route the notebooks through the gateway, set this in `.env`:

```env
LLM_BASE_URL=http://gateway:4000/v1
LLM_API_KEY=<LITELLM_MASTER_KEY>
LLM_MODEL=paper-explainer
LLM_FALLBACK_MODEL=paper-explainer-fallback
EMBED_MODEL=paper-embed
```

When the app calls the gateway, the gateway is the only retry layer. Don't stack retries in both
places. M3 adds Postgres (virtual keys, budgets, spend) and Redis (cache, rate-limit state).

## Checking a real endpoint

```bash
make check-gateway             # uses LLM_* from .env
make check-gateway ARGS=--json # paste-able output
```

`tools/check_gateway.py` reports: chat; `max_tokens` vs `max_completion_tokens`; temperature; seed;
`response_format` (json_schema and json_object); tools (forced and auto); streaming; embeddings;
the status returned for an unknown model; and the model list. It ends with configuration
recommendations, such as setting `LLM_MAX_TOKENS_PARAM=max_completion_tokens`. Every probe is tiny,
so a full run costs well under a cent.

## Notebooks

- Location: `<repo>/notebooks/dayN/` (outside this folder, so GitHub → Colab links stay short).
  Containers mount it at `/notebooks`; override with `NOTEBOOKS_DIR`.
- `notebooks/day1/` holds **byte-for-byte copies** of the reference notebooks, renamed
  `Day1_CodeAlong` → `Practice` and `Day1_Instructor_Demo` → `Demo`. `CHECKSUMS.json` and a test
  guard them. To refresh them: `make notebooks` (reads `../reference/`).
- `make exec-notebooks` runs each Practice notebook four ways, each in a fresh kernel:
  1. **unsolved** (as participants get it): errors are allowed only in TODO-test cells;
  2. **solved** (TODOs filled in): zero errors, and every TODO test prints ✅;
  3. **from each ⏩ catch-up cell** to the end: zero errors;
  4. the **Demo** notebook once.

  Executed copies are saved in `build/executed/`.
- Notebooks are written for Colab (Secrets). The same files run headlessly here (`.env`) with no edits,
  which is how `make exec-notebooks` proves they work before class.

## Package layout

```
paper_agent/
  llm_client.py        Day 1 reliable client (ported verbatim from the SOLUTION cell)
  schemas.py           Answer, KeyTerm, PaperExplainer
  prompts/             versioned prompt files: explainer_v1.txt
  explain.py           paper_messages, check_citations, explain_paper, show
  tokens.py            count tokens, fit sections to a budget, cost
  config.py            load settings from Colab Secrets or the environment
  testing.py           FakeLLM, FlakyProvider, fake_response
  fixtures/tinycoder.py  the Day 1 paper (FICTIONAL, written for the workshop)
mock_llm/              the OpenAI-compatible mock
gateway/               LiteLLM config
tools/                 build_notebooks, exec_notebooks, check_gateway
tests/                 offline tests (fakes + mock-llm)
```

## Troubleshooting

**Port already in use.** Change `MOCK_PORT` or `GATEWAY_PORT` in `.env`. The mock
defaults to host port 8100 because 8000 is often taken. For example, the F5 BIG-IP VPN client listens
on `127.0.0.1:8000`, and `curl localhost:8000` then reaches the VPN instead of the mock.
Find the owner with `lsof -nP -iTCP:<port> -sTCP:LISTEN`.

**Corporate proxy.** If `curl localhost:...` returns a redirect to a login page, your shell's
`HTTP_PROXY` is catching local traffic. Use `curl --noproxy '*' ...`, or add `localhost,127.0.0.1` to
`NO_PROXY`. Containers get `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` from `.env`. Compose service
names are already in the default `NO_PROXY`.

**TLS errors (`CERTIFICATE_VERIFY_FAILED`).** Put your organisation's root CA in `certs/` as a
`.crt` file and run `make build` (see `certs/README.md`).

**Apple Silicon.** Images build natively for arm64. Every dependency, including the LiteLLM image,
ships arm64 builds, so no emulation is needed.
