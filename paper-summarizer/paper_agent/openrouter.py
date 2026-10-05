"""OpenRouter specifics, on top of the plain OpenAI SDK (OpenRouter is OpenAI-compatible).

Configuration is unchanged: LLM_BASE_URL=https://openrouter.ai/api/v1, LLM_API_KEY=<OpenRouter key>,
LLM_MODEL=<vendor/model>, e.g. nvidia/nemotron-3-ultra-550b-a55b:free.

What differs from OpenAI, and what this module helps with:
- Reasoning is a request option, `reasoning`, sent with the SDK's `extra_body`:
      {"enabled": True} | {"enabled": False} | {"effort": "low"|"medium"|"high"} | {"max_tokens": 2000}
- Reasoning tokens are OUTPUT tokens and count toward max_tokens. A small max_tokens on a reasoning
  model can return EMPTY content with finish_reason="length".
- The reply carries `reasoning` (text) and `reasoning_details` (structured). When you send the
  assistant turn back (follow-ups, repairs, tool loops), include `reasoning_details` unmodified.
- Free models (`:free`) allow 20 requests/minute and 50 requests/day without purchased credits
  (1000/day with at least $10 of credits). Over the limit: HTTP 429 with X-RateLimit-* headers.
"""
import os
from typing import Optional

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
APP_URL = "https://mari-muthu-k.github.io/agent-to-production/"
APP_TITLE = "Paper Summarizer Workshop"

REASONING_OFF = {"enabled": False}
REASONING_ON = {"enabled": True}


def is_openrouter(base_url: Optional[str] = None) -> bool:
    return "openrouter.ai" in (base_url or os.environ.get("LLM_BASE_URL") or "")


def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> OpenAI:
    """An OpenAI SDK client for LLM_BASE_URL. On OpenRouter it adds the optional app-attribution
    headers (they label your requests on openrouter.ai; no effect on results)."""
    base_url = base_url or os.environ.get("LLM_BASE_URL")
    kwargs.setdefault("max_retries", 0)          # one retry layer: ours (Day 1)
    if is_openrouter(base_url):
        kwargs["default_headers"] = {"HTTP-Referer": APP_URL, "X-Title": APP_TITLE,
                                     **(kwargs.get("default_headers") or {})}
    return OpenAI(api_key=api_key or os.environ.get("LLM_API_KEY"), base_url=base_url, **kwargs)


def reasoning_body(reasoning: Optional[dict], extra_body: Optional[dict] = None) -> Optional[dict]:
    """extra_body for the SDK with OpenRouter's `reasoning` option merged in."""
    if reasoning is None:
        return extra_body
    return {**(extra_body or {}), "reasoning": reasoning}


def _get(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None and getattr(obj, "model_extra", None):
        value = obj.model_extra.get(name)
    return value


def assistant_turn(message, content: Optional[str] = None) -> dict:
    """The assistant message to send back in the next request, with reasoning_details passed back
    unmodified (OpenRouter needs them to continue the model's reasoning) and tool calls kept."""
    turn = {"role": "assistant", "content": _get(message, "content") if content is None else content}
    details = _get(message, "reasoning_details")
    if details:
        turn["reasoning_details"] = [d if isinstance(d, dict) else d.model_dump() if hasattr(d, "model_dump")
                                     else dict(d) for d in details]
    tool_calls = _get(message, "tool_calls")
    if tool_calls:
        turn["tool_calls"] = [t if isinstance(t, dict) else t.model_dump() for t in tool_calls]
    return turn


def reasoning_text(message) -> Optional[str]:
    return _get(message, "reasoning")


def reasoning_tokens(usage) -> int:
    details = _get(usage, "completion_tokens_details") if usage is not None else None
    return int((_get(details, "reasoning_tokens") if details is not None else 0) or 0)


def rate_limit_message(exc: Exception) -> str:
    """A human sentence for a 429, using OpenRouter's X-RateLimit-* headers when present."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    limit, remaining = headers.get("x-ratelimit-limit"), headers.get("x-ratelimit-remaining")
    msg = "rate limited (HTTP 429)"
    if limit is not None:
        msg += f": limit {limit}, remaining {remaining}"
    return msg + (". Free models allow 20 requests/minute and 50/day without credits "
                  "(1000/day with $10+ of credits).")
