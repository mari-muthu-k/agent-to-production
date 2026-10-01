# FAQ

Common setup problems and how to fix them. Still stuck? Ask in the session chat.

<!-- TODO: Mari — add issues seen during dry runs -->

## Google Colab

??? question "The notebook opens read-only / my changes disappear"
    Notebooks opened from GitHub are temporary copies. Use **File → Save a copy in
    Drive** before you start editing.

??? question "\"Runtime disconnected\" or the notebook stopped responding"
    Free Colab runtimes disconnect after inactivity. Click **Reconnect**, then
    **Runtime → Run all** (or run the cells from the top) to restore your variables.

??? question "I can't sign in to Colab with my work account"
    Some organisations block Colab. Use a personal Google account instead.

## API keys

??? question "`SecretNotFoundError: LLM_API_KEY`"
    The secret name must be exactly `LLM_API_KEY` (case-sensitive). Open the **key
    icon** in Colab's left sidebar and check the spelling. See [Setup](setup.md#4-store-the-key-in-colab-secrets).

??? question "Error about notebook access to the secret"
    In the Secrets panel, turn on the **Notebook access** toggle for `LLM_API_KEY`.
    Each new notebook copy needs access granted once.

??? question "`401 Unauthorized` / invalid API key"
    The key was copied incompletely or has been revoked. Create a new key in your
    provider's dashboard and update the secret.

## Rate limits and quotas

??? question "`429 Too Many Requests` / rate limit exceeded"
    Free tiers allow only a few requests per minute. Wait a minute and retry. On
    Day 1 you will add automatic retries with backoff so your code handles this.

??? question "My daily quota is used up"
    Switch to a different free provider or model for the rest of the day (update the
    `LLM_PROVIDER` secret), or pair with a neighbour.

## Other

??? question "Can I run the labs locally instead of in Colab?"
    Yes, if you are comfortable with Python environments. Set `LLM_API_KEY` as an
    environment variable — the notebook setup cell falls back to it outside Colab.
    We can't help debug local setups during the session.
