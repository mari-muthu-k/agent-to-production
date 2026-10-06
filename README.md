# agent-to-production

Course repository for **Agent to Production - Hands-on**, an 8-hour (4 × 2-hour) hands-on
workshop on building an AI agent and shipping it to production.

**Course site:** <https://mari-muthu-k.github.io/agent-to-production/>

Over four days you build a **Research Paper Summarizer agent with memory**: upload a
research paper (PDF), and the agent parses it, indexes it with embeddings and vector
search (RAG), and explains it in simple terms.

| Folder | Contents |
|--------|----------|
| `docs/` | Source for the course website (MkDocs Material) |
| `labs/` | Google Colab notebooks, one per day; `labs/solutions/` after each session |
| `capstone/` | The final Research Paper Summarizer app |
| `sample_papers/` | Links to open-access arXiv papers to test with |
| `reference-docs/` | Reference material (e.g. LangChain demo) |
| `additional-docs/` | Background reading (e.g. AI Index Report 2026) |

## Participants

Everything runs in Google Colab — nothing to install. Follow the
[setup checklist](https://mari-muthu-k.github.io/agent-to-production/setup/) before Day 1.

The notebooks work with **any OpenAI-compatible provider that has a free tier** (and, from Day 2, an
embedding model). Two that work well:

| Setting (Colab Secret) | What it is | Google Gemini | OpenRouter |
|---|---|---|---|
| `LLM_BASE_URL` | The API's address | `https://generativelanguage.googleapis.com/v1beta/openai` | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | Your key | Google AI Studio → **Get API key** | openrouter.ai → **Keys** (starts `sk-or-`) |
| `LLM_MODEL` | The chat model | A Gemini chat model from the AI Studio model list | A model whose id ends in `:free` |
| `LLM_FALLBACK_MODEL` | Used when the first model is busy or out of its daily quota | A second Gemini chat model | A second `:free` model |
| `EMBED_MODEL` | The embedding model (from Day 2) | A Gemini model whose name contains `embedding` | An embedding model (openrouter.ai/models, filter: embeddings) |

Model names change often: use the ones pinned in the workshop channel. The notebooks never assume a
model name or a vector length; they print what your provider returns.

> ⚠️ **Never put company data into free APIs.** Free tiers may log, keep or train on what you send.
> Use the course's fictional paper or public papers (for example from arXiv), never internal documents.

**Free-tier limits.** Free tiers cap requests per minute and per day, and the caps change often.
For example, on 6 October 2026 a new Gemini project allowed 20 chat requests per day for one model, and
counted every text sent for embedding as one request (100 per minute). So:

- set `LLM_FALLBACK_MODEL`, so a busy or capped model falls back to another;
- rehearse on a **separate key** (or project), so rehearsal doesn't use up the quota you teach with;
- `llm_client` retries per-minute limits for you; a daily cap says so plainly
  ("daily free-tier limit reached: switch to LLM_FALLBACK_MODEL or another key").

To run the notebooks outside Colab, put the same names in a `.env` file (see `.env.example`).

## LLM providers

You don't need a paid subscription to follow along. **[awesome-free-llm-apis](https://github.com/mnfst/awesome-free-llm-apis)** is a community-maintained list of LLM APIs with *permanent* free tiers. Trial credits and time-limited promotions aren't included. It's split into two groups:

- **Provider APIs**: companies serving models they train themselves (e.g. Google Gemini, Mistral AI, Cohere)
- **Inference providers**: platforms hosting open-weight models (e.g. Groq, OpenRouter, NVIDIA NIM, Cloudflare Workers AI)

Some good places to start:

| Provider | Why it's useful for this course |
|----------|---------------------------------|
| Google Gemini | Generous free tier with a large context window, no credit card needed |
| Groq | Very fast inference on open models |
| OpenRouter | Many models behind one API (use models with the `:free` suffix) |
| Mistral AI | Free monthly credits |

> Free-tier limits (rate limits, daily quotas, and which models are included) change often. Check the list for current details before a session.

## Preview the site locally

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdocs serve                     # open http://127.0.0.1:8000/agent-to-production/
```

Pushing to `main` builds the site with `mkdocs build --strict` and deploys it to
GitHub Pages via `.github/workflows/deploy.yml`.
