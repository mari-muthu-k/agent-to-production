"""mock-llm: an OpenAI-compatible server for offline rehearsal, notebooks and tests.

    POST /v1/chat/completions   stream, response_format, tools/tool_choice, stop, seed, max_tokens|max_completion_tokens
    POST /v1/embeddings         deterministic hashed embeddings (float or base64)
    GET  /v1/models             known model names
    GET  /health
    /mock/faults, /mock/stats, /mock/reset   control endpoints (see mock_llm/faults.py)

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

from mock_llm import embeddings, responders, text
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

app = FastAPI(title="mock-llm", version="1.0")
FAULTS = FaultQueue()
STATS: Counter = Counter()

_ERROR_TYPES = {400: ("invalid_request_error", None), 401: ("invalid_request_error", "invalid_api_key"),
                403: ("permission_error", None), 404: ("invalid_request_error", "model_not_found"),
                408: ("timeout", None), 429: ("rate_limit_error", "rate_limit_exceeded")}


def error(status: int, message: str, param: Optional[str] = None, retry_after: Optional[float] = None):
    etype, code = _ERROR_TYPES.get(status, ("server_error", None))
    headers = {"Retry-After": f"{retry_after:g}"} if retry_after is not None else None
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
    rf = body.get("response_format")
    if rf is not None and (not isinstance(rf, dict) or rf.get("type") not in ("text", "json_object", "json_schema")):
        return error(400, "Invalid 'response_format.type'.", param="response_format")
    tc, tools = body.get("tool_choice"), body.get("tools") or []
    if isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name")
        if name not in {(t.get("function") or {}).get("name") for t in tools}:
            return error(400, f"tool_choice names a function that is not in 'tools': {name}", param="tool_choice")
    return None


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

    req = responders.Request(
        messages=body["messages"], model=model, temperature=body.get("temperature"), seed=body.get("seed"),
        response_format=body.get("response_format"), tools=body.get("tools"), tool_choice=body.get("tool_choice"))
    reply = responders.respond(req, fault)
    max_tokens = min(body.get("max_tokens") or DEFAULT_MAX_TOKENS,
                     body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS)

    finish_reason, content, tool_calls = "stop", reply.content, None
    if reply.tool_calls:
        finish_reason = "tool_calls"
        tool_calls = [{"id": c.id, "type": "function",
                       "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in reply.tool_calls]
        out_tokens = sum(text.count(t["function"]["arguments"]) for t in tool_calls) + 3
    else:
        content = text.apply_stop(content, body.get("stop"))
        if fault.truncate:
            max_tokens = min(max_tokens, max(1, text.count(content) // 2))
        content, cut = text.truncate(content, max_tokens)
        finish_reason = "length" if cut else "stop"
        out_tokens = text.count(content)
    usage = {"prompt_tokens": prompt_tokens(body["messages"], body.get("tools")), "completion_tokens": out_tokens}
    usage["total_tokens"] = usage["prompt_tokens"] + out_tokens
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
    data = [{"object": "embedding", "index": i, "embedding": embeddings.encode(embeddings.embed(s, dim), fmt)}
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


@app.post("/mock/reset")
async def reset():
    FAULTS.clear()
    STATS.clear()
    return {"ok": True}
