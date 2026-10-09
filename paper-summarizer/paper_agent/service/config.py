"""Day 4 settings: one place for names, limits and prices. Values come from the environment (.env in Docker)
or Colab Secrets (load_secrets), never from code.

GATEWAY_MODE chooses where LiteLLM runs:
  library  a litellm.Router inside this process (Colab, the jupyter/tests containers). Budgets and rate limits
           are our own few lines of code (usage.py, app.py), kept in process memory: fine for one process.
  proxy    the LiteLLM proxy (`gateway` service, Postgres + Redis) owns retries, fallback, budgets and rate
           limits for every app; each user calls it with their own virtual key (the api service).
"""
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")   # LiteLLM: don't fetch prices from GitHub at import

ALIAS = "paper-explainer"                  # the only model name our code asks for
FALLBACK_ALIAS = ALIAS + "-fallback"
EMBED_ALIAS = "paper-embed"
MAX_TOKENS = 400                           # on every model call: also caps the budget overshoot (5.4)
JUDGE_MAX_TOKENS = 120
NUM_RETRIES = 2                            # LiteLLM's retries: the ONE retry layer
TIMEOUT_S = 30
MAX_UPLOAD_MB = 10
DEFAULT_BUDGET_USD = 1.00                  # per user, per day
WORKSHOP_PRICES = (0.15, 0.60)             # USD per 1M input / output tokens, when no price secret is set

SECRETS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODEL", "EMBED_MODEL",
           "PRICE_IN_PER_M", "PRICE_OUT_PER_M")


def load_secrets(names=SECRETS) -> None:
    """Colab Secrets -> environment variables (an existing variable wins). Outside Colab: nothing to do."""
    try:
        from google.colab import userdata
    except ImportError:
        return
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:                   # secret not defined, or notebook access is off
            continue
        if value and not os.environ.get(name):
            os.environ[name] = value
    global PRICE_IN_PER_M, PRICE_OUT_PER_M
    PRICE_IN_PER_M, PRICE_OUT_PER_M = prices()


def prices() -> tuple:
    """(input, output) USD per 1M tokens. An empty or missing secret means the workshop price."""
    def read(name, default):
        value = (os.environ.get(name) or "").strip()
        return float(value) if value else default
    return read("PRICE_IN_PER_M", WORKSHOP_PRICES[0]), read("PRICE_OUT_PER_M", WORKSHOP_PRICES[1])


PRICE_IN_PER_M, PRICE_OUT_PER_M = prices()


def gateway_mode() -> str:
    mode = os.environ.get("GATEWAY_MODE", "library").strip().lower()
    if mode not in ("library", "proxy"):
        raise ValueError(f"GATEWAY_MODE must be 'library' or 'proxy', not {mode!r}")
    return mode


GATEWAY_MODE = gateway_mode()
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:4000").rstrip("/")
