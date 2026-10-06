"""
llm_client.py - the reliable LLM client you build on Day 1.

Every model call on Days 2, 3 and 4 goes through this module, so each
protection here (timeouts, retries, circuit breaker, fallback, budget,
validation, logging) protects the whole Paper Summarizer agent.

It talks to any OpenAI-compatible HTTP API (OpenRouter, Gemini, ...) directly
with `requests` (no SDK). Settings are read from the environment on EVERY call,
so you can switch models mid-session: LLM_MODEL may be one model or a
comma-separated list ("model-a,model-b"); the next one is used when one is
rate-limited, out of credits or unavailable.

Day 2 adds embed(): embedding calls get the same protections as chat().
"""
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("llm_client")
T = TypeVar("T", bound=BaseModel)
OPENROUTER_URL = "https://openrouter.ai/api/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"   # Gemini's OpenAI-compatible API


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TransientError(Exception):
    """A temporary failure, used for fault injection. Optional retry_after in seconds."""
    def __init__(self, message="transient failure", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class TruncatedOutputError(Exception):
    """The model hit max_tokens before finishing (finish_reason == 'length')."""


class StructuredOutputError(Exception):
    """Output failed validation, even after one repair attempt."""


class CircuitOpenError(Exception):
    """The circuit breaker is open: fail fast instead of calling a failing provider."""


class BudgetExceededError(Exception):
    """This client has spent its cost budget."""


class APIError(Exception):
    """An HTTP error from the API. `status_code`, the error `body` and the `response` are attached."""
    def __init__(self, message, status_code=None, response=None, body=None):
        super().__init__(message)
        self.message, self.status_code, self.response, self.body = message, status_code, response, body


class BadRequestError(APIError): pass          # 400: invalid request (bad parameter, context too long)
class AuthenticationError(APIError): pass      # 401: wrong or missing key
class PaymentRequiredError(APIError): pass     # 402: out of credits (OpenRouter)
class PermissionDeniedError(APIError): pass    # 403: e.g. input flagged by moderation
class NotFoundError(APIError): pass            # 404: unknown model or URL
class RateLimitError(APIError): pass           # 429: too many requests
class DailyQuotaError(RateLimitError): pass    # 429 that will not clear today: a free tier's daily cap
class ServerError(APIError): pass              # 408, 5xx: timeout upstream, provider down, no provider
class APITimeoutError(APIError): pass          # we gave up waiting
class APIConnectionError(APIError): pass       # network failure


STATUS_ERRORS = {400: BadRequestError, 401: AuthenticationError, 402: PaymentRequiredError,
                 403: PermissionDeniedError, 404: NotFoundError, 408: ServerError, 429: RateLimitError}

# Worth retrying: the same request may succeed a moment later.
RETRYABLE_ERRORS = (
    APITimeoutError,       # took too long
    APIConnectionError,    # network blip
    RateLimitError,        # HTTP 429
    ServerError,           # HTTP 408, 5xx
    TransientError,
)
# NOT retryable: they fail identically every time, and you pay for each try.
#   BadRequestError (400), AuthenticationError (401), PaymentRequiredError (402),
#   PermissionDeniedError (403), NotFoundError (404)


DAILY_CAP_MESSAGE = "daily free-tier limit reached: switch to LLM_FALLBACK_MODEL or another key"


def _seconds(value) -> float:
    """'31387s' or 31387 -> 31387.0; anything else -> 0."""
    try:
        return float(str(value).rstrip("s"))
    except (TypeError, ValueError):
        return 0.0


def is_daily_cap(err: dict) -> bool:
    """A 429 body that names a per-day quota, or asks us to wait more than two minutes."""
    for detail in err.get("details") or []:
        if not isinstance(detail, dict):
            continue
        if any("perday" in str(v.get("quotaId", "")).lower() for v in detail.get("violations") or []):
            return True
        if _seconds(detail.get("retryDelay")) > 120:
            return True
    return False


def api_error(response, body) -> APIError:
    """Turn an error response into the matching exception class.
    Some providers (Gemini) send the error object inside a JSON list: [{"error": {...}}]."""
    if isinstance(body, list) and body and isinstance(body[0], dict):
        body = body[0]
    err = body.get("error", {}) if isinstance(body, dict) else {}
    err = err if isinstance(err, dict) else {"message": str(err)}
    code = err.get("code")
    status = response.status_code if response.status_code >= 400 else (code if isinstance(code, int) else 500)
    message = err.get("message") or (response.text or "")[:300] or f"HTTP {status}"
    raw = (err.get("metadata") or {}).get("raw")
    if raw:
        message += f" (provider said: {str(raw)[:200]})"
    if status == 429 and is_daily_cap(err):
        return DailyQuotaError(f"HTTP 429: {DAILY_CAP_MESSAGE} (provider said: {message[:200]})",
                               status_code=status, response=response, body=body)
    cls = STATUS_ERRORS.get(status, ServerError if status >= 500 else APIError)
    exc = cls(f"HTTP {status}: {message}", status_code=status, response=response, body=body)
    delays = [_seconds(d.get("retryDelay")) for d in err.get("details") or [] if isinstance(d, dict)]
    if any(delays):                       # Gemini puts the wait in the body ("retryDelay": "20s"), not a header
        exc.retry_after = max(delays)
    return exc


GEMINI_MODEL = re.compile(r"(^|/)(models/)?gemini", re.IGNORECASE)


def reasoning_effort(model: Optional[str]) -> Optional[str]:
    """The `reasoning_effort` to send: LLM_REASONING_EFFORT if set ("" or "default" = send nothing),
    else "none" for Gemini models, whose thinking otherwise eats max_tokens and truncates answers."""
    value = os.environ.get("LLM_REASONING_EFFORT")
    if value is not None:
        value = value.strip().lower()
        return None if value in ("", "default") else value
    return "none" if model and GEMINI_MODEL.search(model) else None


# ---------------------------------------------------------------------------
# The HTTP client: OpenRouter's REST API with requests
# ---------------------------------------------------------------------------
class OpenRouterClient:
    """POSTs JSON to any OpenAI-compatible URL (OpenRouter, Gemini, ...) and returns the JSON reply as a dict.

    The key and URL come from LLM_API_KEY / LLM_BASE_URL at call time unless given here.
    With GEMINI_API_KEY set, Google URLs use that key (and LLM_BASE_URL defaults to Gemini).
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 60.0, headers: Optional[dict] = None):
        self._api_key, self._base_url, self.timeout = api_key, base_url, timeout
        self.extra_headers = headers or {}
        self._effort_for: dict = {}       # model -> the reasoning_effort it accepted

    @property
    def api_key(self) -> str:
        return self._api_key or os.environ.get("LLM_API_KEY", "")

    @property
    def base_url(self) -> str:
        default = GEMINI_URL if os.environ.get("GEMINI_API_KEY") else OPENROUTER_URL
        return (self._base_url or os.environ.get("LLM_BASE_URL") or default).rstrip("/")

    def auth_headers(self) -> dict:
        """Google URL + GEMINI_API_KEY: send it as x-goog-api-key, and as Bearer too, because Gemini's
        OpenAI-compatible endpoint requires the Authorization header. Otherwise: Bearer LLM_API_KEY."""
        gemini = os.environ.get("GEMINI_API_KEY")
        if gemini and not self._api_key and "googleapis.com" in self.base_url:
            return {"x-goog-api-key": gemini, "Authorization": f"Bearer {gemini}"}
        return {"Authorization": f"Bearer {self.api_key}"}

    def headers(self, extra: Optional[dict] = None) -> dict:
        return {**self.auth_headers(), "Content-Type": "application/json",
                "HTTP-Referer": "https://mari-muthu-k.github.io/agent-to-production/",   # optional app attribution
                "X-Title": "Paper Summarizer Workshop", **self.extra_headers, **(extra or {})}

    def post(self, path: str, body: dict, headers: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        try:
            r = requests.post(f"{self.base_url}/{path}", json=body, headers=self.headers(headers),
                              timeout=timeout or self.timeout)
        except requests.Timeout as e:
            raise APITimeoutError(f"no answer within {timeout or self.timeout}s") from e
        except requests.ConnectionError as e:
            raise APIConnectionError(f"could not reach {self.base_url}: {e}") from e
        try:
            data = r.json()
        except ValueError:
            data = None
        # OpenRouter can also answer 200 with an error body (e.g. the provider failed mid-request).
        if r.status_code >= 400 or not isinstance(data, dict) or ("error" in data and "choices" not in data):
            raise api_error(r, data)
        return data

    def is_google(self) -> bool:
        return "googleapis.com" in self.base_url

    def is_openrouter(self) -> bool:
        return "openrouter.ai" in self.base_url

    def chat(self, headers: Optional[dict] = None, timeout: Optional[float] = None, **body) -> dict:
        """POST /chat/completions. `body` is the JSON request: model, messages, max_tokens, ..."""
        if self.is_google():
            body.pop("reasoning", None)   # OpenRouter's switch; Gemini uses reasoning_effort
        model = body.get("model")
        effort = reasoning_effort(model)
        if effort and "reasoning_effort" not in body:
            if self.is_openrouter():      # OpenRouter's own form of the same setting
                body.setdefault("reasoning", {"enabled": False} if effort == "none" else {"effort": effort})
            else:                         # the OpenAI-compatible parameter (Gemini, OpenAI, ...)
                body["reasoning_effort"] = self._effort_for.get(model, effort)
                try:
                    return self.post("chat/completions", body, headers, timeout)
                except BadRequestError:
                    # Some models reject "none" (Gemini's lite models do): retry once with "low", and remember it.
                    if body["reasoning_effort"] != "none" or os.environ.get("LLM_REASONING_EFFORT"):
                        raise
                    body["reasoning_effort"] = self._effort_for[model] = "low"
        return self.post("chat/completions", body, headers, timeout)

    def embeddings(self, headers: Optional[dict] = None, timeout: Optional[float] = None, **body) -> dict:
        """POST /embeddings with {"model": ..., "input": [...]}."""
        return self.post("embeddings", body, headers, timeout)


def env_models(name: str = "LLM_MODEL") -> list:
    """Models listed in an environment variable: "a" or "a,b,c" (read now, so switching takes effect)."""
    return [m.strip() for m in os.environ.get(name, "").split(",") if m.strip()]


# ---------------------------------------------------------------------------
# Config and records
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    model: str = ""                       # "" = LLM_MODEL, read at every call; may be "a,b,c"
    price_in_per_1m: float = 0.0          # USD per 1M input tokens (0 for :free models)
    price_out_per_1m: float = 0.0         # USD per 1M output tokens
    timeout_s: float = 30.0
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    fallback_model: Optional[str] = None  # tried once each if the primary fails ("" = LLM_FALLBACK_MODEL)
    budget_usd: Optional[float] = None    # refuse new calls once this much is spent
    max_concurrency: int = 4              # client-side limit on parallel calls
    breaker_threshold: int = 5            # consecutive failures before the circuit opens
    breaker_reset_s: float = 30.0         # how long the circuit stays open
    embed_model: str = ""                 # "" = EMBED_MODEL, read at every call
    embed_batch_size: int = 100           # inputs per /embeddings request (providers cap this)
    max_tokens_param: str = "max_tokens"  # some models want "max_completion_tokens"
    reasoning: Optional[dict] = None      # OpenRouter, e.g. {"enabled": False} or {"effort": "low"}; None = default


@dataclass
class CallRecord:
    request_id: str
    model: str
    status: str               # ok | truncated | failed | fallback_ok
    attempts: int
    input_tokens: Optional[int]     # None when the provider reported no usage (see usage_reported)
    output_tokens: Optional[int]
    latency_ms: int
    cost_usd: float
    finish_reason: Optional[str]
    reasoning_tokens: int = 0   # "thinking" tokens: billed as output, and they count toward max_tokens
    usage_reported: bool = True     # False: the provider sent no token counts, so cost is unknown, not 0
    kind: str = "chat"              # chat | embed


CALL_LOG: list = []   # every call lands here; Day 4 ships it to real tracing


def _strip_to_json(text: str) -> str:
    """Remove ```json fences and chatter around the JSON object."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """closed -> (too many failures) -> open -> (after a pause) -> half_open -> one trial call."""

    def __init__(self, threshold: int = 5, reset_s: float = 30.0):
        self.threshold, self.reset_s = threshold, reset_s
        self.failures = 0
        self.opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= self.reset_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self.failures, self.opened_at = 0, None

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.threshold:
            self.opened_at = time.monotonic()
            logger.warning(f"circuit OPEN after {self.failures} consecutive failures")


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None, client=None, fallback_client=None):
        self.config = config or LLMConfig()
        # One retry layer only: ours, so we can see, control and log every attempt.
        self.client = client or OpenRouterClient(timeout=self.config.timeout_s)
        self.fallback_client = fallback_client or self.client
        self.breaker = CircuitBreaker(self.config.breaker_threshold, self.config.breaker_reset_s)
        # Embeddings get their own circuit (same rules): an embedding outage must not block chat, or vice versa.
        self.embed_breaker = CircuitBreaker(self.config.breaker_threshold, self.config.breaker_reset_s)
        self._slots = threading.BoundedSemaphore(self.config.max_concurrency)
        self._lock = threading.Lock()
        self._turns = threading.local()   # each thread's last assistant turn, for repairs and follow-ups
        self.exhausted: set = set()       # models out of daily quota or credits: skipped from now on
        self.spent_usd = 0.0

    @property
    def last_message(self) -> Optional[dict]:
        """The last assistant turn (with OpenRouter's reasoning_details) to send back in a follow-up."""
        return getattr(self._turns, "message", None)

    def models(self) -> list:
        """Models to try, in order: primary first, then fallbacks; exhausted ones are skipped."""
        split = lambda value: [m.strip() for m in value.split(",") if m.strip()]   # noqa: E731
        fallbacks = split(self.config.fallback_model or "")
        if self.config.model:                   # models set in code: use exactly those
            primary = split(self.config.model)
        else:                                   # models from the environment, re-read on every call
            primary, fallbacks = env_models("LLM_MODEL"), fallbacks or env_models("LLM_FALLBACK_MODEL")
        # A fallback keeps its own attempt even if it has the primary's name: the fallback client may be
        # another provider, and with no LLM_FALLBACK_MODEL the course reuses the primary's name on purpose.
        ordered = list(dict.fromkeys(primary)) + list(dict.fromkeys(fallbacks))
        usable = [m for m in ordered if m not in self.exhausted]
        if not ordered:
            raise ValueError("no model configured: set LLM_MODEL (one model, or a comma-separated list)")
        return usable or ordered[-1:]    # everything exhausted: still try the last one (it may have reset)

    # -- helpers ------------------------------------------------------------
    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter: attempt 0 -> up to 1s, 1 -> 2s, 2 -> 4s ...
        Jitter stops thousands of clients retrying at the same instant (a thundering herd)."""
        exp = min(self.config.max_delay_s, self.config.base_delay_s * (2 ** attempt))
        return random.uniform(0, exp)

    def _retry_after(self, exc: Exception) -> float:
        """Seconds the server asked us to wait (HTTP Retry-After header), if any."""
        value = getattr(exc, "retry_after", None)
        if value is None:
            headers = getattr(getattr(exc, "response", None), "headers", None) or {}
            value = headers.get("retry-after")
        try:
            return min(float(value), self.config.max_delay_s) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _quota_gone(self, exc: Exception) -> bool:
        """Out of credits (402), or a 429 that will not clear soon (e.g. a free model's daily cap).
        Waiting is pointless: switch to the next model instead."""
        if isinstance(exc, (PaymentRequiredError, DailyQuotaError)):
            return True
        if not isinstance(exc, RateLimitError):
            return False
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        try:
            reset_s = float(headers.get("x-ratelimit-reset", 0)) / 1000 - time.time()   # epoch milliseconds
            wait = float(headers.get("retry-after") or 0)
        except (TypeError, ValueError):
            return False
        return max(reset_s, wait) > 120

    def _cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok * self.config.price_in_per_1m + out_tok * self.config.price_out_per_1m) / 1_000_000

    def _record(self, **kwargs) -> CallRecord:
        rec = CallRecord(**kwargs)
        with self._lock:
            CALL_LOG.append(rec)
            self.spent_usd += rec.cost_usd
        logger.info(json.dumps(asdict(rec)))   # one JSON line per call: easy to ship to Datadog/ELK
        return rec

    def _daily_cap(self, exc: Exception, model: str) -> Optional[DailyQuotaError]:
        """A plain-language error for a 429 that will not clear today (detected from headers), else None."""
        if isinstance(exc, (DailyQuotaError, PaymentRequiredError)) or not self._quota_gone(exc):
            return None
        return DailyQuotaError(f"{DAILY_CAP_MESSAGE} ({model}: {exc})", status_code=getattr(exc, "status_code", 429),
                               response=getattr(exc, "response", None), body=getattr(exc, "body", None))

    def _reraise(self, exc: Exception, model: str):
        plain = self._daily_cap(exc, model)
        if plain is None:
            raise exc
        raise plain from exc

    def _check_budget(self) -> None:
        if self.config.budget_usd is not None and self.spent_usd >= self.config.budget_usd:
            raise BudgetExceededError(f"spent ${self.spent_usd:.6f} of ${self.config.budget_usd:.6f} budget")

    def _create_with_retries(self, model: str, params: dict, request_id: str, call=None, breaker=None):
        """Call one model with retries. Returns (response, attempts). `call` defaults to chat."""
        call, breaker = call or self.client.chat, breaker or self.breaker
        attempts = 0
        while True:
            if not breaker.allow():
                raise CircuitOpenError("provider is failing; circuit open, failing fast")
            attempts += 1
            try:
                resp = call(model=model, **params)
                breaker.record_success()
                return resp, attempts
            except RETRYABLE_ERRORS as exc:
                breaker.record_failure()
                exc.attempts = attempts
                if attempts > self.config.max_retries or self._quota_gone(exc):
                    raise
                delay = max(self._backoff_delay(attempts - 1), self._retry_after(exc))
                logger.warning(f"[{request_id}] attempt {attempts} failed ({type(exc).__name__}); "
                               f"retrying in {delay:.2f}s")
                time.sleep(delay)

    # -- main API -----------------------------------------------------------
    def chat(self, messages: list, max_tokens: int = 500,
             temperature: Optional[float] = 0.0, **kwargs) -> str:
        self._check_budget()

        request_id = uuid.uuid4().hex[:8]
        params = {"messages": messages, self.config.max_tokens_param: max_tokens, **kwargs}
        if temperature is not None:   # some reasoning models reject temperature
            params["temperature"] = temperature
        if self.config.reasoning is not None:   # OpenRouter's reasoning switch, part of the JSON body
            params["reasoning"] = self.config.reasoning

        start = time.perf_counter()
        primary, *others = self.models()
        model_used, status = primary, "ok"
        with self._slots:             # client-side concurrency limit
            try:
                resp, attempts = self._create_with_retries(primary, params, request_id)
            except RETRYABLE_ERRORS + (CircuitOpenError, PaymentRequiredError) as exc:
                if self._quota_gone(exc):
                    self.exhausted.add(primary)
                    logger.warning(f"[{request_id}] {primary} is out of quota; skipping it from now on")
                if not others:
                    self._record(request_id=request_id, model=model_used, status="failed", attempts=0,
                                 input_tokens=0, output_tokens=0, finish_reason=None, cost_usd=0.0,
                                 latency_ms=int((time.perf_counter() - start) * 1000))
                    self._reraise(exc, primary)
                resp = None
                for model_used in others:          # each fallback model gets one attempt
                    logger.warning(f"[{request_id}] {type(exc).__name__} on {primary}; trying {model_used}")
                    try:
                        resp = self.fallback_client.chat(model=model_used, **params)
                        break
                    except RETRYABLE_ERRORS + (PaymentRequiredError,) as fallback_exc:
                        if self._quota_gone(fallback_exc):
                            self.exhausted.add(model_used)
                        exc = fallback_exc
                if resp is None:
                    self._record(request_id=request_id, model=model_used, status="failed", attempts=0,
                                 input_tokens=0, output_tokens=0, finish_reason=None, cost_usd=0.0,
                                 latency_ms=int((time.perf_counter() - start) * 1000))
                    self._reraise(exc, model_used)
                status, attempts = "fallback_ok", 1

        choice, usage = resp["choices"][0], resp.get("usage") or {}
        message = choice.get("message") or {}
        reported = usage.get("prompt_tokens") is not None
        in_tok = usage.get("prompt_tokens") if reported else None
        out_tok = (usage.get("completion_tokens") or 0) if reported else None
        reasoning_tok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        truncated = choice.get("finish_reason") == "length"
        self._record(request_id=request_id, model=model_used,
                     status="truncated" if truncated else status, attempts=attempts,
                     input_tokens=in_tok, output_tokens=out_tok,
                     latency_ms=int((time.perf_counter() - start) * 1000),
                     cost_usd=round(self._cost(in_tok or 0, out_tok or 0), 6),
                     finish_reason=choice.get("finish_reason"),
                     reasoning_tokens=reasoning_tok, usage_reported=reported)

        # Keep the assistant turn for follow-ups. OpenRouter: pass reasoning_details back unmodified.
        turn = {"role": "assistant", "content": message.get("content")}
        if message.get("reasoning_details"):
            turn["reasoning_details"] = message["reasoning_details"]
        self._turns.message = turn
        hint = ("; the model's reasoning used the whole budget: raise max_tokens or set LLM_REASONING_EFFORT=none"
                if reasoning_tok and not message.get("content") else "")

        # Never hand half an answer to the rest of the system.
        if truncated:
            raise TruncatedOutputError(f"[{request_id}] output cut off at max_tokens={max_tokens}{hint}")
        return message.get("content")

    def chat_structured(self, messages: list, schema: Type[T], max_tokens: int = 500, **kwargs) -> T:
        """Return a validated Pydantic object. On failure, show the model its error and retry once."""
        text = self.chat(messages, max_tokens=max_tokens, **kwargs)
        try:
            return schema.model_validate_json(_strip_to_json(text))
        except ValidationError as first_error:
            logger.warning(f"validation failed ({first_error.error_count()} errors); repairing once")
            repair = messages + [
                self.last_message or {"role": "assistant", "content": text},
                {"role": "user", "content": "Your previous reply failed validation:\n"
                                            f"{first_error}\nReply with ONLY the corrected JSON object."},
            ]
            text2 = self.chat(repair, max_tokens=max_tokens, **kwargs)
            try:
                return schema.model_validate_json(_strip_to_json(text2))
            except ValidationError as second_error:
                raise StructuredOutputError(str(second_error)) from second_error


    # -- embeddings (Day 2) -------------------------------------------------
    def embed(self, texts, model: Optional[str] = None, dimensions: Optional[int] = None) -> list:
        """One vector (a list of floats) per text, in order. Texts go in batched requests
        (embed_batch_size per request), each with the same protections as chat(): budget guard,
        concurrency cap, a circuit breaker (its own), retries with backoff, daily-cap detection, timeout_s,
        and one log line per request (output_tokens is 0: embeddings have no output tokens).
        Vectors are returned exactly as the provider sent them: not every provider normalises."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        model = model or self.config.embed_model or os.environ.get("EMBED_MODEL", "").strip()
        if not model:
            raise ValueError("no embedding model configured: set EMBED_MODEL")
        vectors = []
        for i in range(0, len(texts), self.config.embed_batch_size):
            vectors += self._embed_batch(model, texts[i:i + self.config.embed_batch_size], dimensions)
        return vectors

    def _embed_batch(self, model: str, batch: list, dimensions: Optional[int]) -> list:
        self._check_budget()
        request_id, start = uuid.uuid4().hex[:8], time.perf_counter()
        params = {"input": batch, "timeout": self.config.timeout_s}
        if dimensions:
            params["dimensions"] = dimensions
        with self._slots:             # client-side concurrency limit
            try:
                resp, attempts = self._create_with_retries(model, params, request_id, call=self.client.embeddings,
                                                           breaker=self.embed_breaker)
                data = sorted(resp.get("data") or [], key=lambda d: d.get("index", 0))
                if len(data) != len(batch):
                    raise APIError(f"asked for {len(batch)} embeddings, got {len(data)}", body=resp)
            except Exception as exc:
                self._record(request_id=request_id, model=model, status="failed",
                             attempts=getattr(exc, "attempts", 1), input_tokens=0, output_tokens=0,
                             latency_ms=int((time.perf_counter() - start) * 1000), cost_usd=0.0,
                             finish_reason=None, kind="embed")
                self._reraise(exc, model)
        usage = resp.get("usage") or {}
        reported = usage.get("prompt_tokens") is not None
        in_tok = usage.get("prompt_tokens") if reported else None
        self._record(request_id=request_id, model=model, status="ok", attempts=attempts,
                     input_tokens=in_tok, output_tokens=0,
                     latency_ms=int((time.perf_counter() - start) * 1000),
                     cost_usd=round(self._cost(in_tok or 0, 0), 6), finish_reason=None,
                     usage_reported=reported, kind="embed")
        return [d["embedding"] for d in data]


def summarize_calls(log: Optional[list] = None) -> dict:
    """Totals across recorded calls: the seed of cost and latency monitoring."""
    log = CALL_LOG if log is None else log
    if not log:
        return {"calls": 0}
    lat = sorted(r.latency_ms for r in log)
    return {
        "calls": len(log),
        "by_status": {s: sum(r.status == s for r in log) for s in sorted({r.status for r in log})},
        "by_model": {m: sum(r.model == m for r in log) for m in sorted({r.model for r in log})},
        "retries": sum(max(r.attempts - 1, 0) for r in log),
        "input_tokens": sum(r.input_tokens or 0 for r in log),
        "output_tokens": sum(r.output_tokens or 0 for r in log),
        "usage_unreported": sum(not r.usage_reported for r in log),
        "total_cost_usd": round(sum(r.cost_usd for r in log), 6),
        "p50_latency_ms": lat[len(lat) // 2],
        "max_latency_ms": lat[-1],
    }
