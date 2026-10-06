"""OpenRouter helpers around llm_client.OpenRouterClient (plain HTTP with `requests`, no SDK).

Configuration (unchanged names, read at call time, so they can change mid-session):
    LLM_BASE_URL   https://openrouter.ai/api/v1
    LLM_API_KEY    your OpenRouter key (sk-or-...)
    LLM_MODEL      one model, or a comma-separated list tried in order, e.g.
                   "nvidia/nemotron-3-ultra-550b-a55b:free,meta-llama/llama-3.3-70b-instruct:free"
    LLM_FALLBACK_MODEL, EMBED_MODEL

Free models (`:free`): 20 requests/minute, and 50 requests/day per account without purchased credits
(1000/day with $10+ of credits). The daily cap is per account, so switching models does not reset it;
switching does help when one free model is congested ("rate-limited upstream") or offline.

Reasoning: `"reasoning": {"enabled": True|False}` or `{"effort": "low"}` in the JSON body. Reasoning tokens
are output tokens and count toward max_tokens. Send `reasoning_details` back unmodified in follow-ups.
"""
import json
import os
from collections.abc import Iterator
from typing import Optional

import requests

from paper_agent.llm_client import (  # noqa: F401  (re-exported: one import for notebooks)
    DAILY_CAP_MESSAGE,
    OPENROUTER_URL,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    DailyQuotaError,
    NotFoundError,
    OpenRouterClient,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    api_error,
    env_models,
)

REASONING_OFF = {"enabled": False}
REASONING_ON = {"enabled": True}


def is_openrouter(base_url: Optional[str] = None) -> bool:
    return "openrouter.ai" in (base_url or os.environ.get("LLM_BASE_URL") or "")


def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> OpenRouterClient:
    return OpenRouterClient(api_key=api_key, base_url=base_url, **kwargs)


# --- switching models -------------------------------------------------------------
def use_model(model: str, fallback: Optional[str] = None) -> list:
    """Switch the chat model(s) for everything that reads LLM_MODEL, from now on (no restart).
    `model` may be a comma-separated list; returns the list now in use."""
    os.environ["LLM_MODEL"] = model
    if fallback is not None:
        os.environ["LLM_FALLBACK_MODEL"] = fallback
    return current_models()


def current_models() -> list:
    return env_models("LLM_MODEL") + [m for m in env_models("LLM_FALLBACK_MODEL") if m not in env_models("LLM_MODEL")]


def list_models(client: Optional[OpenRouterClient] = None, free_only: bool = True, needs: tuple = (),
                embeddings: bool = False) -> list:
    """Models from GET /models (public, no key needed), optionally only `:free` ones supporting every
    parameter in `needs` (e.g. ("tools",) or ("response_format",)). Most context first."""
    client = client or OpenRouterClient()
    url = f"{client.base_url}/models" + ("?output_modalities=embeddings" if embeddings else "")
    r = requests.get(url, headers=client.headers(), timeout=client.timeout)
    if r.status_code >= 400:
        raise api_error(r, r.json() if r.content else None)
    rows = []
    for m in r.json().get("data", []):
        params = set(m.get("supported_parameters") or [])
        if free_only and is_openrouter(client.base_url) and not m["id"].endswith(":free"):
            continue
        if needs and not set(needs) <= params:
            continue
        rows.append({"id": m["id"], "context_length": m.get("context_length"), "parameters": sorted(params)})
    return sorted(rows, key=lambda m: -(m["context_length"] or 0))


def free_models(needs: tuple = (), client: Optional[OpenRouterClient] = None) -> list:
    """IDs of free chat models that support `needs`; pick one and call use_model(...)."""
    return [m["id"] for m in list_models(client, free_only=True, needs=needs)]


def key_info(client: Optional[OpenRouterClient] = None) -> dict:
    """GET /key: credits used, limit and whether this is a free-tier key."""
    client = client or OpenRouterClient()
    r = requests.get(f"{client.base_url}/key", headers=client.headers(), timeout=client.timeout)
    if r.status_code >= 400:
        raise api_error(r, r.json() if r.content else None)
    return r.json().get("data", {})


# --- requests and replies -----------------------------------------------------------
def stream_chat(client: Optional[OpenRouterClient] = None, **body) -> Iterator[dict]:
    """POST /chat/completions with "stream": true and yield each server-sent event (a dict).
    OpenRouter also sends ': OPENROUTER PROCESSING' keep-alive comments; they are skipped."""
    client = client or OpenRouterClient()
    with requests.post(f"{client.base_url}/chat/completions", json={**body, "stream": True},
                       headers=client.headers(), timeout=client.timeout, stream=True) as r:
        if r.status_code >= 400:
            raise api_error(r, r.json() if r.content else None)
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                return
            event = json.loads(data)
            if "error" in event:
                raise APIError(f"stream error: {event['error'].get('message')}", body=event)
            yield event


def text_of(reply: dict) -> Optional[str]:
    """The assistant's text from a /chat/completions reply."""
    return reply["choices"][0]["message"].get("content")


def assistant_turn(message: dict, content: Optional[str] = None) -> dict:
    """The assistant message to send back in the next request: reasoning_details passed back unmodified
    (OpenRouter needs them to continue the model's reasoning) and tool calls kept."""
    turn = {"role": "assistant", "content": message.get("content") if content is None else content}
    for key in ("reasoning_details", "tool_calls"):
        if message.get(key):
            turn[key] = message[key]
    return turn


def reasoning_text(message: dict) -> Optional[str]:
    return message.get("reasoning")


def reasoning_tokens(usage: Optional[dict]) -> int:
    return int(((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)


def rate_limit_message(exc: Exception) -> str:
    """A human sentence for a 429, provider-neutral; uses X-RateLimit-* headers when present."""
    if isinstance(exc, DailyQuotaError):
        return f"{DAILY_CAP_MESSAGE} (use_model(...) switches the chat model without a restart)"
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    limit, remaining = headers.get("x-ratelimit-limit"), headers.get("x-ratelimit-remaining")
    msg = "rate limited (HTTP 429)"
    if limit is not None:
        msg += f": limit {limit}, remaining {remaining}"
    return msg + (". Free tiers limit requests per minute and per day. llm_client already retried; "
                  "wait 30 seconds, or try another model with use_model(...).")
