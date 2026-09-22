# agent-to-production

Demo monorepo for **Agent to Production**, an 8-hour hands-on course on building AI agents and shipping them to production.

| Folder | Contents |
|--------|----------|
| `session-1` … `sesssion-4` | Code and material for each day |
| `research-papers/` | Source papers and plain-language explainers |
| `reference-docs/` | Reference material (e.g. LangChain demo) |
| `additional-docs/` | Background reading (e.g. AI Index Report 2026) |

## Setup

Copy `.env.example` to `.env` and fill in your provider and API key:

```env
LLM_API_KEY=your_api_key_here
LLM_PROVIDER=openai # anthropic, google, openai, azure
```

## LLM Providers list

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
