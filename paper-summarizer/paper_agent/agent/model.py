"""The chat model for the agent (Day 3, 3.1): ChatOpenAI pointed at the same endpoint as Days 1-2.

LLM_BASE_URL alone chooses the provider (mock-llm offline, a gateway or provider online): no code
changes. Keys come from the environment (Colab Secrets or .env), never from code.
"""
import os

from langchain_openai import ChatOpenAI

MAX_TOKENS = 1000


def make_chat_model(cache=None, **overrides) -> ChatOpenAI:
    """ChatOpenAI from LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (the first model, if it is a list).
    cache: e.g. InMemoryCache() for an exact-match response cache on this model only (6.2)."""
    settings = dict(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"].split(",")[0].strip(),
        temperature=0,
        timeout=30,
        max_tokens=MAX_TOKENS,
        # One retry layer (the Day 1 rule): the client retries twice; no ModelRetryMiddleware on top.
        # Day 4 moves retries to the LiteLLM gateway and sets this to 0.
        max_retries=2,
        cache=cache,
    )
    if os.environ.get("LLM_MAX_TOKENS_PARAM") == "max_tokens":   # ChatOpenAI sends max_completion_tokens;
        settings["max_tokens"] = None                            # some endpoints only accept max_tokens
        settings["extra_body"] = {"max_tokens": MAX_TOKENS}
    settings.update(overrides)
    return ChatOpenAI(**settings)
