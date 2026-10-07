"""mock-llm: an OpenAI-compatible server for offline rehearsal, notebooks and tests.

    POST /v1/chat/completions   stream, response_format, tools/tool_choice, stop, seed, max_tokens|max_completion_tokens
    POST /v1/embeddings         deterministic hashed embeddings (float or base64)
    GET  /v1/models             known model names
    GET  /health
    /mock/faults, /mock/stats, /mock/reset   control endpoints (see mock_llm/faults.py)
    /mock/mode                               Day 3 agent mode: injection obey|resist, force_citation (mock_llm/agent.py)

Usage reports prompt_tokens_details.cached_tokens when a request repeats a long prefix of the previous
one (>= 1,024 tokens, in blocks of 128), like providers with automatic prompt caching (Day 3, 6.5).

Lenient by default, so switching LLM_BASE_URL is the only change between offline and online:
any API key and model name work, except deliberately bad ones (patterns below) used in error demos.
"""
import asyncio
import json
import os
import re
import time
import uuid
from collections import Counter
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mock_llm import agent, embeddings, responders, text
from mock_llm.faults import Fault, FaultQueue, fault_from_headers, merge

REJECT_KEYS = re.compile(os.environ.get("MOCK_REJECT_KEYS", r"wrong|invalid|revoked|expired"), re.I)
UNKNOWN_MODELS = re.compile(os.environ.get("MOCK_UNKNOWN_MODELS", r"^(no-such|unknown|does-not-exist|nonexistent)"))
BROKEN_MODELS = re.compile(os.environ.get("MOCK_BROKEN_MODELS", r"^broken"))
STRICT = os.environ.get("MOCK_STRICT", "0") == "1"
API_KEYS = {k.strip() for k in os.environ.get("MOCK_API_KEYS", "sk-mock-local").split(",") if k.strip()}
MODELS = [m.strip() for m in os.environ.get("MOCK_MODELS", "mock-llm,mock-llm-fallback,mock-embed").split(",")]
BASE_LATENCY_S = float(os.environ.get("MOCK_BASE_LATENCY_MS", "20")) / 1000
TOKEN_LATENCY_S = float(os.environ.get("MOCK_TOKEN_LATENCY_MS", "2")) / 1000
DEFAULT_MAX_TOKENS = 4096
CONTEXT_TOKENS = int(os.environ.get("MOCK_CONTEXT_TOKENS", "16384"))   # like a 16k-context model

app = FastAPI(title="mock-llm", version="1.0")
FAULTS = FaultQueue()
STATS: Counter = Counter()
MODE = agent.Mode()
PREFIX_MIN_TOKENS, PREFIX_BLOCK = 1024, 128
_LAST_PROMPT = {"text": ""}

_ERROR_TYPES = {400: ("invalid_request_error", None), 401: ("invalid_request_error", "invalid_api_key"),
                403: ("permission_error", None), 404: ("invalid_request_error", "model_not_found"),
                408: ("timeout", None), 429: ("rate_limit_error", "rate_limit_exceeded")}


def error(status: int, message: str, param: Optional[str] = None, retry_after: Optional[float] = None):
    etype, code = _ERROR_TYPES.get(status, ("server_error", None))
    headers = {"Retry-After": f"{retry_after:g}"} if retry_after is not None else {}
    if status == 429:      # OpenRouter-style rate-limit headers (free models: 20/min, 50/day)
        reset_ms = int((time.time() + (retry_after or 60)) * 1000)
        headers.update({"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset_ms)})
    STATS[f"status_{status}"] += 1
    return JSONResponse({"error": {"message": message, "type": etype, "param": param, "code": code}},
                        status_code=status, headers=headers)


def check_auth(request: Request):
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not key or REJECT_KEYS.search(key) or (STRICT and key not in API_KEYS):
        shown = f"{key[:3]}...{key[-4:]}" if len(key) > 8 else "***"
        return error(401, f"Incorrect API key provided: {shown}.")
    return None


def check_model(model: Optional[str]):
    if not model:
        return error(400, "you must provide a model parameter", param="model")
    if UNKNOWN_MODELS.search(model) or (STRICT and model not in MODELS):
        return error(404, f"The model `{model}` does not exist or you do not have access to it.")
    if BROKEN_MODELS.search(model):
        return error(500, f"The model `{model}` is failing (mock: broken model).")
    return None


async def read_json(request: Request):
    try:
        body = await request.json()
    except Exception:
        return None, error(400, "We could not parse the JSON body of your request.")
    if not isinstance(body, dict):
        return None, error(400, "The request body must be a JSON object.")
    return body, None


def prompt_tokens(messages: list, tools: Optional[list]) -> int:
    n = 3 + sum(4 + text.count(responders.text_of(m)) for m in messages)
    return n + (text.count(json.dumps(tools)) if tools else 0)


def cached_prefix_tokens(messages: list, tools: Optional[list]) -> int:
    """Tokens of the longest prefix shared with the previous request, if long enough to be cached."""
    prompt = json.dumps(tools or []) + "".join(f"<{m.get('role')}>{responders.text_of(m)}" for m in messages)
    previous, _LAST_PROMPT["text"] = _LAST_PROMPT["text"], prompt
    n = 0
    for a, b in zip(prompt, previous, strict=False):
        if a != b:
            break
        n += 1
    shared = text.count(prompt[:n])
    return 0 if shared < PREFIX_MIN_TOKENS else shared // PREFIX_BLOCK * PREFIX_BLOCK


def mode_for(headers) -> "agent.Mode":
    """The /mock/mode settings, overridden per request by X-Mock-Injection / X-Mock-Force-Citation."""
    injection = (headers.get("x-mock-injection") or MODE.injection).strip().lower()
    return agent.Mode(injection=injection if injection in ("obey", "resist") else MODE.injection,
                      force_citation=headers.get("x-mock-force-citation") or MODE.force_citation)


def validate_chat(body: dict, fault: Fault):
    for p in fault.reject_params:
        if p in body:
            alt = {"max_tokens": " Use 'max_completion_tokens' instead."}.get(p, "")
            return error(400, f"Unsupported parameter: '{p}' is not supported with this model.{alt}", param=p)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return error(400, "'messages' must be a non-empty array.", param="messages")
    if any(not isinstance(m, dict) or "role" not in m for m in messages):
        return error(400, "Each message needs a 'role'.", param="messages")
    t = body.get("temperature")
    if t is not None and not (isinstance(t, int | float) and 0 <= t <= 2):
        return error(400, f"Invalid 'temperature': {t}. Expected a value between 0 and 2.", param="temperature")
    for p in ("max_tokens", "max_completion_tokens"):
        v = body.get(p)
        if v is not None and (not isinstance(v, int) or v < 1):
            return error(400, f"Invalid '{p}': must be a positive integer.", param=p)
    r = body.get("reasoning")
    if r is not None and not isinstance(r, dict):
        return error(400, "'reasoning' must be an object, e.g. {\"enabled\": true}.", param="reasoning")
    rf = body.get("response_format")
    if rf is not None and (not isinstance(rf, dict) or rf.get("type") not in ("text", "json_object", "json_schema")):
        return error(400, "Invalid 'response_format.type'.", param="response_format")
    tc, tools = body.get("tool_choice"), body.get("tools") or []
    if isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name")
        if name not in {(t.get("function") or {}).get("name") for t in tools}:
            return error(400, f"tool_choice names a function that is not in 'tools': {name}", param="tool_choice")
    return None


def reasoning_requested(body: dict) -> tuple:
    """(on?, exclude?) from OpenRouter's `reasoning` object or `reasoning_effort`."""
    r = body.get("reasoning") or {}
    effort = r.get("effort") or body.get("reasoning_effort")
    on = bool(r.get("enabled") or r.get("max_tokens") or (effort and effort != "none"))
    if r.get("enabled") is False or effort == "none":
        on = False
    return on, bool(r.get("exclude"))


def _completion_id() -> str:
    return "chatcmpl-mock" + uuid.uuid4().hex[:20]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "mock"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    STATS["chat_requests"] += 1
    if (resp := check_auth(request)) is not None:
        return resp
    body, err = await read_json(request)
    if err is not None:
        return err
    model = body.get("model")
    fault = merge(fault_from_headers(request.headers), FAULTS.take("chat", model))
    if (resp := check_model(model)) is not None:
        return resp
    if fault.delay_ms:
        await asyncio.sleep(fault.delay_ms / 1000)
    if fault.status:
        return error(fault.status, fault.message or f"mock-injected HTTP {fault.status}", retry_after=fault.retry_after)
    if (resp := validate_chat(body, fault)) is not None:
        return resp

    n_prompt = prompt_tokens(body["messages"], body.get("tools"))
    if n_prompt > CONTEXT_TOKENS:
        STATS["status_400"] += 1
        return JSONResponse({"error": {
            "message": f"This model's maximum context length is {CONTEXT_TOKENS} tokens. However, your messages "
                       f"resulted in {n_prompt} tokens. Please reduce the length of the messages.",
            "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}},
            status_code=400)

    req = responders.Request(
        messages=body["messages"], model=model, temperature=body.get("temperature"), seed=body.get("seed"),
        response_format=body.get("response_format"), tools=body.get("tools"), tool_choice=body.get("tool_choice"))
    thinking, hide_thinking = reasoning_requested(body)
    req.reasoning = thinking
    if agent.is_agent_request(req):
        reply = agent.respond(req, fault, mode_for(request.headers))
    else:
        reply = responders.respond(req, fault)
    max_tokens = min(body.get("max_tokens") or DEFAULT_MAX_TOKENS,
                     body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS)

    # Reasoning (OpenRouter) comes first and is paid from the same max_tokens budget as the answer.
    reasoning, reasoning_tokens = None, 0
    if thinking:
        reasoning, _ = text.truncate(responders.reasoning_for(req), max_tokens)
        reasoning_tokens = text.count(reasoning)
    budget = max_tokens - reasoning_tokens

    finish_reason, content, tool_calls, out_tokens = "stop", reply.content, None, 0
    if budget <= 0:                       # thinking used everything: empty answer, cut off
        finish_reason, content = "length", ""
    elif reply.tool_calls:
        finish_reason = "tool_calls"
        tool_calls = [{"id": c.id, "type": "function",
                       "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in reply.tool_calls]
        out_tokens = sum(text.count(t["function"]["arguments"]) for t in tool_calls) + 3
    else:
        content = text.apply_stop(content, body.get("stop"))
        if fault.truncate:
            budget = min(budget, max(1, text.count(content) // 2))
        content, cut = text.truncate(content, budget)
        finish_reason = "length" if cut else "stop"
        out_tokens = text.count(content)
    out_tokens += reasoning_tokens
    usage = {"prompt_tokens": n_prompt, "completion_tokens": out_tokens}
    usage["total_tokens"] = usage["prompt_tokens"] + out_tokens
    usage["prompt_tokens_details"] = {"cached_tokens": min(cached_prefix_tokens(body["messages"], body.get("tools")),
                                                           n_prompt)}
    if thinking:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    STATS["status_200"] += 1
    STATS["prompt_tokens"] += usage["prompt_tokens"]
    STATS["completion_tokens"] += out_tokens

    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(_stream(model, content, tool_calls, finish_reason, usage, include_usage),
                                 media_type="text/event-stream")

    await asyncio.sleep(BASE_LATENCY_S + TOKEN_LATENCY_S * out_tokens)
    message = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning and not hide_thinking:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{"type": "reasoning.text", "text": reasoning, "format": "unknown", "index": 0}]
    return {"id": _completion_id(), "object": "chat.completion", "created": int(time.time()), "model": model,
            "system_fingerprint": "fp_mock_llm",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
            "usage": usage}


async def _stream(model, content, tool_calls, finish_reason, usage, include_usage):
    cid, created = _completion_id(), int(time.time())

    def chunk(delta, finish=None, usage_=None, choices=True):
        data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "system_fingerprint": "fp_mock_llm",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}] if choices else []}
        if usage_ is not None:
            data["usage"] = usage_
        return f"data: {json.dumps(data)}\n\n"

    await asyncio.sleep(BASE_LATENCY_S)
    yield chunk({"role": "assistant", "content": ""})
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield chunk({"tool_calls": [{"index": i, **tc}]})
    else:
        for piece in text.pieces(content):
            await asyncio.sleep(TOKEN_LATENCY_S)
            yield chunk({"content": piece})
    yield chunk({}, finish=finish_reason)
    if include_usage:
        yield chunk(None, usage_=usage, choices=False)
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    STATS["embedding_requests"] += 1
    if (resp := check_auth(request)) is not None:
        return resp
    body, err = await read_json(request)
    if err is not None:
        return err
    model = body.get("model")
    fault = merge(fault_from_headers(request.headers), FAULTS.take("embeddings", model))
    if (resp := check_model(model)) is not None:
        return resp
    if fault.delay_ms:
        await asyncio.sleep(fault.delay_ms / 1000)
    if fault.status:
        return error(fault.status, fault.message or f"mock-injected HTTP {fault.status}", retry_after=fault.retry_after)
    raw = body.get("input")
    if isinstance(raw, str):
        inputs = [raw]
    elif isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
        inputs = raw
    elif isinstance(raw, list) and raw and all(isinstance(x, int) for x in raw):
        inputs = [" ".join(map(str, raw))]                     # one pre-tokenized input
    elif isinstance(raw, list) and raw and all(isinstance(x, list) for x in raw):
        inputs = [" ".join(map(str, x)) for x in raw]
    else:
        return error(400, "'input' must be a non-empty string or array.", param="input")
    if any(not s.strip() for s in inputs):
        return error(400, "'input' cannot contain empty strings.", param="input")
    dim = body.get("dimensions") or embeddings.DEFAULT_DIM
    if not isinstance(dim, int) or not 1 <= dim <= 4096:
        return error(400, "Invalid 'dimensions'.", param="dimensions")
    fmt = body.get("encoding_format") or "float"
    data = [{"object": "embedding", "index": i, "embedding": embeddings.encode(embeddings.embed(s, dim, model), fmt)}
            for i, s in enumerate(inputs)]
    n = sum(text.count(s) for s in inputs)
    STATS["status_200"] += 1
    STATS["embedding_tokens"] += n
    await asyncio.sleep(BASE_LATENCY_S)
    return {"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": n, "total_tokens": n}}


# ---------------------------------------------------------------------------
# Control endpoints
# ---------------------------------------------------------------------------
@app.post("/mock/faults")
async def add_fault(fault: Fault):
    FAULTS.add(fault)
    return {"queued": FAULTS.list()}


@app.get("/mock/faults")
async def get_faults():
    return {"queued": FAULTS.list()}


@app.delete("/mock/faults")
async def clear_faults():
    FAULTS.clear()
    return {"queued": []}


@app.get("/mock/stats")
async def stats():
    return dict(STATS)


@app.get("/mock/mode")
async def get_mode():
    return MODE.__dict__


@app.post("/mock/mode")
async def set_mode(update: dict):
    """{"injection": "obey" | "resist", "force_citation": "c99" | null}; omitted keys stay as they are."""
    if update.get("injection", MODE.injection) not in ("obey", "resist"):
        return error(400, "injection must be 'obey' or 'resist'", param="injection")
    for key in ("injection", "force_citation"):
        if key in update:
            setattr(MODE, key, update[key])
    return MODE.__dict__


@app.post("/mock/reset")
async def reset():
    FAULTS.clear()
    STATS.clear()
    MODE.injection, MODE.force_citation = "obey", None
    _LAST_PROMPT["text"] = ""
    return {"ok": True}
