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

To run notebooks locally instead, copy `.env.example` to `.env` and fill in your
provider and API key:

```env
LLM_API_KEY=your_api_key_here
LLM_PROVIDER=openai # anthropic, google, openai, azure
```

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
