# Pre-workshop setup

Please finish this checklist **before Day 1**. It takes about 15 minutes. If anything
fails, check the [FAQ](faq.md).

## 1. Google account

You need a Google account to use Google Colab. A personal account works fine.

- [ ] I can sign in at <https://accounts.google.com>

## 2. Test Google Colab

1. Open <https://colab.research.google.com> and click **New notebook**.
2. In the first cell, type the code below and press ++shift+enter++.

    ```python
    import sys
    print("Hello from Colab!", sys.version)
    ```

- [ ] I see `Hello from Colab!` and a Python version printed below the cell.

## 3. Get an LLM API key

The labs work with **any OpenAI-compatible provider that has a free tier and an embedding model**.
Free tiers are enough for this course. Two that work well:

| Setting (Colab Secret) | What it is | Google Gemini | OpenRouter |
|---|---|---|---|
| `LLM_BASE_URL` | The API's address | `https://generativelanguage.googleapis.com/v1beta/openai` | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | Your key | Google AI Studio → **Get API key** | openrouter.ai → **Keys** (starts `sk-or-`) |
| `LLM_MODEL` | The chat model | A Gemini chat model from the AI Studio model list | A model whose id ends in `:free` |
| `LLM_FALLBACK_MODEL` | Used when the first model is busy or out of its daily quota | A second Gemini chat model | A second `:free` model |
| `EMBED_MODEL` | The embedding model (from Day 2) | A Gemini model whose name contains `embedding` | An embedding model (openrouter.ai/models, filter: embeddings) |

Model names change often: use the ones pinned in the workshop channel. The notebooks never assume a
model name or a vector length; they print what your provider returns.

!!! danger "No company data"
    **Never put company data into free APIs.** Free tiers may log, keep or train on what you send.
    Use the course's fictional paper or public papers (for example from arXiv), never internal documents.

1. Sign up with the provider you chose.
2. Create an API key in its dashboard.
3. Copy the key somewhere safe for the next step.

!!! warning "Production"
    Treat an API key like a password. Never paste it into a notebook cell, a
    screenshot, chat, or a Git commit. Anyone with the key can spend your quota.

- [ ] I have an API key.

## 4. Store the key in Colab Secrets

Colab Secrets keep your key out of the notebook so it is never saved or shared with it.

1. In any Colab notebook, click the **key icon** (Secrets) in the left sidebar.
2. Click **Add new secret**.
3. Set **Name** to `LLM_API_KEY` and **Value** to your key.
4. Turn on the **Notebook access** toggle.
5. Add `LLM_BASE_URL`, `LLM_MODEL`, `LLM_FALLBACK_MODEL` and (for Day 2) `EMBED_MODEL` the same way,
   with the values from the table above, and turn on **Notebook access** for each.

Test it in a cell:

```python
from google.colab import userdata

key = userdata.get("LLM_API_KEY")
print("Key loaded:", key[:4] + "…")
```

- [ ] The cell prints `Key loaded:` followed by the first characters of my key.

??? success "Solution"
    Seeing `SecretNotFoundError`? The secret name must be exactly `LLM_API_KEY`
    (case-sensitive). Seeing a notebook-access error? Turn on the **Notebook access**
    toggle next to the secret, then run the cell again.

## 5. Free-tier limits

**Free-tier limits.** Free tiers cap requests per minute and per day, and the caps change often.
For example, on 6 October 2026 a new Gemini project allowed 20 chat requests per day for one model, and
counted every text sent for embedding as one request (100 per minute). So:

- set `LLM_FALLBACK_MODEL`, so a busy or capped model falls back to another;
- rehearse on a **separate key** (or project), so rehearsal doesn't use up the quota you teach with;
- `llm_client` retries per-minute limits for you; a daily cap says so plainly
  ("daily free-tier limit reached: switch to LLM_FALLBACK_MODEL or another key").

## You're ready

- [ ] All boxes above are ticked. See you on Day 1!
