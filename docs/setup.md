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

The labs work with any of the supported providers. Free tiers are enough for this
course — see the repository README for a list of providers with free tiers.

<!-- TODO: Mari — confirm the recommended provider and model for the labs -->

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
5. (Optional) Add a second secret `LLM_PROVIDER` with a value such as `google`,
   `openai`, or `anthropic`.

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

## You're ready

- [ ] All boxes above are ticked. See you on Day 1!
