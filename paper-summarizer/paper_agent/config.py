"""Where configuration comes from: Colab Secrets in class, environment variables (.env) locally.

Keys are never hardcoded. Offline vs online is chosen by LLM_BASE_URL alone:
    http://mock-llm:8000/v1   offline mock (Docker)
    https://<your gateway>/v1 real models
"""
import os
from typing import Optional

from paper_agent.llm_client import GEMINI_URL, LLMClient, LLMConfig

REQUIRED = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
OPTIONAL = ("LLM_FALLBACK_MODEL", "EMBED_MODEL", "LLM_MAX_TOKENS_PARAM", "GEMINI_API_KEY")


def load_env(required=REQUIRED, optional=OPTIONAL) -> dict:
    """Copy Colab Secrets into os.environ when running in Colab; otherwise use the environment as is.

    GEMINI_API_KEY can stand in for LLM_API_KEY: with only a Gemini key, LLM_BASE_URL defaults to
    Gemini's OpenAI-compatible endpoint. Raises a readable error listing what is missing.
    """
    optional = (*optional, "GEMINI_API_KEY") if "GEMINI_API_KEY" not in optional else optional
    try:
        from google.colab import userdata  # type: ignore
        for name in (*required, *optional):
            try:
                value = userdata.get(name)
                if value:
                    os.environ[name] = value
            except Exception:   # secret not defined or notebook access off
                pass
    except ImportError:
        pass
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = GEMINI_URL
    has_key = bool(os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    missing = [n for n in required if not os.environ.get(n) and not (n == "LLM_API_KEY" and has_key)]
    if missing:
        missing = [("LLM_API_KEY (or GEMINI_API_KEY)" if n == "LLM_API_KEY" else n) for n in missing]
        raise RuntimeError(f"missing configuration: {', '.join(missing)}. "
                           "Set them in Colab Secrets, or in .env when running in Docker.")
    return {n: os.environ.get(n) for n in (*required, *optional)}


def llm_config_from_env(price_in_per_1m: float = 0.0, price_out_per_1m: float = 0.0, **overrides) -> LLMConfig:
    env = load_env()
    fields = dict(
        model="",                 # "" = read LLM_MODEL / LLM_FALLBACK_MODEL at every call (switchable)
        price_in_per_1m=price_in_per_1m,
        price_out_per_1m=price_out_per_1m,
        max_tokens_param=env.get("LLM_MAX_TOKENS_PARAM") or "max_tokens",
    )
    fields.update(overrides)
    return LLMConfig(**fields)


def llm_client_from_env(config: Optional[LLMConfig] = None, **overrides) -> LLMClient:
    """An LLMClient for the configured endpoint; LLM_* variables are re-read on every call."""
    return LLMClient(config or llm_config_from_env(**overrides))
