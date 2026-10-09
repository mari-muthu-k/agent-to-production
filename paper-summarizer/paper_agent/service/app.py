"""The service (Day 4, 4.2): four endpoints around the Day 3 agent.

    POST /papers                upload a PDF -> paper_id (a hash of the content)
    POST /papers/{id}/ask       question -> JSON answer, citations, usage
    POST /papers/{id}/explain   the same run as server-sent events (progress, answer, done)
    GET  /me/usage              my calls, spend and budget

Every request carries an API key (Authorization: Bearer ...), mapped to a user. Known errors become clear
statuses with a sentence a person can read: 401 unknown key, 413 too large, 415 not a PDF, 429 budget or
rate limit, 503 provider down. Never a stack trace. One JSON log line per request, with the X-Request-ID
the response carries.

`uvicorn paper_agent.service.app:app` serves build_app() (created on first access, from the environment).
"""
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from paper_agent.service import config
from paper_agent.service.errors import BudgetExceededError, RateLimitedError
from paper_agent.service.gateway import CURRENT_KEY
from paper_agent.service.pipeline import EXPLAIN_Q, ask_paper, build_agent, explain_events
from paper_agent.service.store import PaperStore
from paper_agent.service.usage import usage_of
from paper_agent.service.validate import validate_upload

# Library mode: demo keys -> users. At work this is your single sign-on. Proxy mode: the key is the user's
# LiteLLM virtual key, and the proxy says whose it is (GET /key/info).
API_KEYS = {"key-alice": "alice", "key-bob": "bob", "key-eval": "eval-bot"}
# Library mode: requests per minute per user, counted in this process (proxy mode: the key's rpm_limit).
RPM_LIMITS: dict = {}
HITS: dict = defaultdict(deque)
LOG = logging.getLogger("paper_agent.requests")
LOG.addHandler(logging.NullHandler())          # silent until someone attaches a handler (6.1)
LOG.setLevel(logging.INFO)
LOG.propagate = False
ERROR_LOG = logging.getLogger("paper_agent.errors")
KEY_USERS: dict = {}                          # proxy mode: virtual key -> user id (a cache of /key/info)


class Question(BaseModel):
    question: str
    prompt_version: Optional[str] = None       # the eval runner compares prompt versions (7.5, `make eval`)


def error_body(error: str, message: str) -> dict:
    return {"error": error, "message": message}


# --- who is calling -------------------------------------------------------------------------------
def key_info(key: str) -> dict:
    """The proxy's record for a virtual key (proxy mode). Raises 401 for a key the proxy doesn't know."""
    try:
        r = httpx.get(f"{config.GATEWAY_URL}/key/info", params={"key": key},
                      headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except httpx.HTTPError:
        raise HTTPException(503, "The gateway is unavailable. Try again in a minute.")
    if r.status_code != 200:
        raise HTTPException(401, "Unknown API key.")
    return r.json().get("info") or {}


def current_user(request: Request, authorization: Optional[str] = Header(None)) -> str:
    key = (authorization or "").removeprefix("Bearer ").strip()
    if config.GATEWAY_MODE == "proxy" and key:
        if key not in KEY_USERS:
            KEY_USERS[key] = key_info(key).get("user_id") or "unknown"
        user = KEY_USERS[key]
    else:
        user = API_KEYS.get(key)
    if not user:
        raise HTTPException(401, "Unknown API key.")
    request.state.user, request.state.api_key = user, key if config.GATEWAY_MODE == "proxy" else None
    check_rate_limit(user)
    return user


def check_rate_limit(user_id: str) -> None:
    """Library mode: at most RPM_LIMITS[user] requests in any 60 seconds."""
    limit = RPM_LIMITS.get(user_id)
    if not limit or config.GATEWAY_MODE == "proxy":
        return
    now, hits = time.monotonic(), HITS[user_id]
    while hits and now - hits[0] >= 60:
        hits.popleft()
    if len(hits) >= limit:
        raise RateLimitedError(user_id, retry_after=math.ceil(60 - (now - hits[0])), limit=limit)
    hits.append(now)


def set_rate_limit(user_id: str, rpm: Optional[int]) -> None:
    """5.5: set (or with None, remove) a user's limit; the count starts now."""
    HITS.pop(user_id, None)
    if rpm:
        RPM_LIMITS[user_id] = rpm
    else:
        RPM_LIMITS.pop(user_id, None)


# --- the app ----------------------------------------------------------------------------------------
def build_app(agent=None, store: Optional[PaperStore] = None) -> FastAPI:
    quiet = {"/health"}                    # health checks every few seconds: no spans, no log lines
    app = FastAPI(title="Paper explainer", version="4.0",
                  telemetry={"exclude": lambda scope: scope.get("path") in quiet})
    app.state.agents = {}
    app.state.agent = agent
    app.state.store = store or PaperStore(os.environ.get("DATA_DIR", "."))

    def agent_for(version: Optional[str]):
        if version is None or version == getattr(app.state.agent, "prompt_version", None):
            if app.state.agent is None:
                app.state.agent = build_agent()
            return app.state.agent
        if version not in ("v1", "v2"):
            raise HTTPException(422, "prompt_version must be v1 or v2.")
        if version not in app.state.agents:
            app.state.agents[version] = build_agent(version, extra_middleware=extra_middleware())
        return app.state.agents[version]

    def paper_or_404(paper_id: str) -> None:
        if not app.state.store.get(paper_id):
            raise HTTPException(404, f"No paper {paper_id}: upload it first (POST /papers).")

    @app.get("/health")
    def health():
        return {"status": "ok", "gateway_mode": config.GATEWAY_MODE}

    @app.post("/papers")
    def upload(request: Request, file: UploadFile = File(...), user: str = Depends(current_user)):  # noqa: B008
        data = file.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)    # never read more than the limit + 1
        validate_upload(data)
        CURRENT_KEY.set(request.state.api_key)          # proxy mode: the uploader's key pays for the embeddings
        paper_id, is_new = app.state.store.ingest(data)
        chunks = len(app.state.store.get(paper_id)["chunks"])
        return {"paper_id": paper_id, "filename": file.filename, "chunks": chunks,
                "status": "processed" if is_new else "already processed"}

    @app.post("/papers/{paper_id}/ask")
    def ask(paper_id: str, body: Question, request: Request, user: str = Depends(current_user)):
        paper_or_404(paper_id)
        result = ask_paper(paper_id, body.question, user, agent=agent_for(body.prompt_version),
                           api_key=request.state.api_key)
        request.state.usage = result
        return {"paper_id": paper_id, **result}

    @app.post("/papers/{paper_id}/explain")
    def explain(paper_id: str, request: Request, body: Optional[Question] = None, user: str = Depends(current_user)):
        paper_or_404(paper_id)
        events: queue.Queue = queue.Queue()
        agent = agent_for(body.prompt_version if body else None)

        def run():               # one thread for the whole run, so its spans open and close in one context
            try:
                for event, data in explain_events(paper_id, body.question if body else EXPLAIN_Q, user, agent,
                                                  request.state.api_key):
                    if event == "done":
                        request.state.usage = data
                    events.put((event, data))
            except Exception as exc:                       # mid-stream: the status is already 200
                status, payload = error_response(exc)
                events.put(("error", {**json.loads(payload.body), "status": status}))
            events.put(None)

        threading.Thread(target=run, daemon=True).start()

        def stream():
            while (item := events.get()) is not None:
                yield f"event: {item[0]}\ndata: {json.dumps(item[1])}\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/me/usage")
    def my_usage(request: Request, user: str = Depends(current_user)):
        if config.GATEWAY_MODE == "proxy":
            info = key_info(request.state.api_key)
            spent, budget = info.get("spend") or 0.0, info.get("max_budget")
            return {"user": user, "llm_calls": None, "spent_usd": round(spent, 6), "budget_usd": budget,
                    "remaining_usd": None if budget is None else round(budget - spent, 6), "source": "gateway"}
        return usage_of(user)

    # --- errors: a status and a sentence, never a stack trace ----------------------------------------
    @app.exception_handler(BudgetExceededError)
    @app.exception_handler(RateLimitedError)
    def too_much(request, exc):
        return error_response(exc)[1]

    @app.exception_handler(HTTPException)
    def http_error(request, exc: HTTPException):
        slug = {401: "unauthorized", 404: "not_found", 413: "too_large", 415: "unsupported_type",
                422: "invalid_request", 503: "gateway_unavailable"}.get(exc.status_code, "error")
        return JSONResponse(error_body(slug, exc.detail), status_code=exc.status_code)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        """One JSON line per request, written when the response has been sent. The question, the answer and
        the paper's text are never logged."""
        request_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:                            # anything not handled above
            status, response = error_response(exc)
        response.headers["X-Request-ID"] = request_id
        body = response.body_iterator if hasattr(response, "body_iterator") else None

        def write():
            usage = getattr(request.state, "usage", None) or {}
            LOG.info(json.dumps({
                "request_id": request_id, "user": getattr(request.state, "user", None),
                "route": f"{request.method} {request.url.path}", "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "llm_calls": usage.get("llm_calls", 0), "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0), "cost_usd": usage.get("cost_usd", 0.0),
                "answered_by": usage.get("answered_by", [])}))

        if request.url.path in quiet:
            return response
        if body is None:
            write()
            return response

        async def logged():
            async for chunk in body:
                yield chunk
            write()
        response.body_iterator = logged()
        return response

    return app


def extra_middleware() -> list:
    """Library mode: our budget check before every model call. Proxy mode: the proxy checks budgets."""
    from paper_agent.service.usage import budget_guard
    return [budget_guard] if config.GATEWAY_MODE == "library" else []


def error_response(exc: BaseException) -> tuple:
    """(status, JSONResponse) for an exception that escaped a route."""
    import litellm
    import openai
    if isinstance(exc, BudgetExceededError):
        return 429, JSONResponse(error_body("budget_exceeded", (
            f"You've used your budget for today (${exc.spent:.6f} of ${exc.budget:.6f}). It resets at midnight "
            "UTC; ask your admin if you need more.")), status_code=429)
    if isinstance(exc, RateLimitedError):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else {}
        wait = f"Try again in {exc.retry_after} seconds." if exc.retry_after else "Try again in a minute."
        limit = f" ({exc.limit} per minute)" if exc.limit else ""
        return 429, JSONResponse(error_body("rate_limited", f"Too many requests{limit}. {wait}"),
                                 status_code=429, headers=headers)
    if isinstance(exc, (litellm.exceptions.APIError, litellm.exceptions.ServiceUnavailableError,
                        litellm.exceptions.APIConnectionError, litellm.exceptions.Timeout,
                        litellm.exceptions.InternalServerError, openai.APIConnectionError, openai.InternalServerError)):
        ERROR_LOG.warning("provider error %s: %s", type(exc).__name__, str(exc)[:200])
        return 503, JSONResponse(error_body("provider_unavailable", "The model provider is unavailable right now. "
                                                                    "Try again in a minute."), status_code=503)
    ERROR_LOG.error("unhandled %s: %s", type(exc).__name__, str(exc)[:300])
    return 500, JSONResponse(error_body("internal_error", "Something went wrong on our side."), status_code=500)


class LogCapture(logging.Handler):
    """6.1: every request's log line, printed and kept, so a request id can be looked up."""

    def __init__(self, echo: bool = True):
        super().__init__()
        self.lines, self.echo = [], echo

    def emit(self, record):
        line = record.getMessage()
        self.lines.append(json.loads(line))
        if self.echo:
            print(line)

    def find(self, request_id: str, wait_s: float = 2.0) -> Optional[dict]:
        """The log line of one request (it is written just after the response, so wait a moment)."""
        deadline = time.monotonic() + wait_s
        while True:
            found = next((line for line in self.lines if line["request_id"] == request_id), None)
            if found or time.monotonic() > deadline:
                return found
            time.sleep(0.02)


def attach_log_handler(handler: Optional[logging.Handler] = None) -> logging.Handler:
    """Send the request log lines somewhere (one handler at a time): a LogCapture, or stdout."""
    for old in [h for h in LOG.handlers if not isinstance(h, logging.NullHandler)]:
        LOG.removeHandler(old)
    handler = handler or LogCapture()
    LOG.addHandler(handler)
    return handler


SERVERS: dict = {}


def start_server(app: FastAPI, port: int = 8765) -> str:
    """4.4: serve `app` with uvicorn in a background thread of this notebook. Running it again stops the old
    server first, so the cell (and every catch-up cell) can be re-run."""
    import uvicorn
    if port in SERVERS:
        old, thread = SERVERS.pop(port)
        old.should_exit = True
        thread.join(timeout=10)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError(f"the server could not start: is port {port} in use? Runtime → Restart session, "
                               "then run 5.0")
        time.sleep(0.05)
    SERVERS[port] = (server, thread)
    return f"http://127.0.0.1:{port}"


def create_app() -> FastAPI:
    """The api service: everything from the environment (.env). GATEWAY_MODE=proxy in Docker."""
    from paper_agent.service.gateway import make_router
    from paper_agent.service.store import setup_retrieval
    from paper_agent.service.tracing import setup_tracing
    data_dir = os.environ.get("DATA_DIR", ".")
    router = make_router() if config.GATEWAY_MODE == "library" else None
    setup_retrieval(router, data_dir)
    if os.environ.get("TRACING", "console") != "off":
        setup_tracing()
    if os.environ.get("LOG_REQUESTS", "1") == "1":
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        attach_log_handler(handler)
    ERROR_LOG.addHandler(logging.StreamHandler())          # the type and a short message; never a stack trace
    agent = build_agent(os.environ.get("PROMPT_VERSION", "v1"), extra_middleware=extra_middleware(), router=router)
    return build_app(agent, PaperStore(data_dir))


def __getattr__(name):                  # `app` is built on first access, so importing this module is cheap
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
