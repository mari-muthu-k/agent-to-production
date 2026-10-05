"""
llm_client.py - the reliable LLM client you build on Day 1.

Every model call on Days 2, 3 and 4 goes through this module, so each
protection here (timeouts, retries, circuit breaker, fallback, budget,
validation, logging) protects the whole Paper Summarizer agent.
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

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("llm_client")
T = TypeVar("T", bound=BaseModel)


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


# Worth retrying: the same request may succeed a moment later.
RETRYABLE_ERRORS = (
    openai.APITimeoutError,       # took too long
    openai.APIConnectionError,    # network blip
    openai.RateLimitError,        # HTTP 429
    openai.InternalServerError,   # HTTP 5xx
    TransientError,
)
# NOT retryable: they fail identically every time, and you pay for each try.
#   openai.BadRequestError (400), AuthenticationError (401),
#   PermissionDeniedError (403), NotFoundError (404)


# ---------------------------------------------------------------------------
# Config and records
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    model: str
    price_in_per_1m: float = 0.0          # USD per 1M input tokens
    price_out_per_1m: float = 0.0         # USD per 1M output tokens
    timeout_s: float = 30.0
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    fallback_model: Optional[str] = None  # tried once if the primary fails
    budget_usd: Optional[float] = None    # refuse new calls once this much is spent
    max_concurrency: int = 4              # client-side limit on parallel calls
    breaker_threshold: int = 5            # consecutive failures before the circuit opens
    breaker_reset_s: float = 30.0         # how long the circuit stays open
    max_tokens_param: str = "max_tokens"  # some models want "max_completion_tokens"
    reasoning: Optional[dict] = None      # OpenRouter, e.g. {"enabled": False} or {"effort": "low"}; None = default


@dataclass
class CallRecord:
    request_id: str
    model: str
    status: str               # ok | truncated | failed | fallback_ok
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    finish_reason: Optional[str]
    reasoning_tokens: int = 0   # "thinking" tokens: billed as output, and they count toward max_tokens


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
    def __init__(self, config: LLMConfig, client=None, fallback_client=None):
        self.config = config
        # max_retries=0: we do our own retries so we can see, control and log them.
        # Never stack SDK retries with your own: 3 x 3 = 9 attempts.
        self.client = client or OpenAI(
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL") or None,
            timeout=config.timeout_s,
            max_retries=0,
        )
        self.fallback_client = fallback_client or self.client
        self.breaker = CircuitBreaker(config.breaker_threshold, config.breaker_reset_s)
        self._slots = threading.BoundedSemaphore(config.max_concurrency)
        self._lock = threading.Lock()
        self._turns = threading.local()   # each thread's last assistant turn, for repairs and follow-ups
        self.spent_usd = 0.0

    @property
    def last_message(self) -> Optional[dict]:
        """The last assistant turn (with OpenRouter's reasoning_details) to send back in a follow-up."""
        return getattr(self._turns, "message", None)

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

    def _cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok * self.config.price_in_per_1m + out_tok * self.config.price_out_per_1m) / 1_000_000

    def _record(self, **kwargs) -> CallRecord:
        rec = CallRecord(**kwargs)
        with self._lock:
            CALL_LOG.append(rec)
            self.spent_usd += rec.cost_usd
        logger.info(json.dumps(asdict(rec)))   # one JSON line per call: easy to ship to Datadog/ELK
        return rec

    def _create_with_retries(self, params: dict, request_id: str):
        """Call the primary model with retries. Returns (response, attempts)."""
        attempts = 0
        while True:
            if not self.breaker.allow():
                raise CircuitOpenError("provider is failing; circuit open, failing fast")
            attempts += 1
            try:
                resp = self.client.chat.completions.create(model=self.config.model, **params)
                self.breaker.record_success()
                return resp, attempts
            except RETRYABLE_ERRORS as exc:
                self.breaker.record_failure()
                if attempts > self.config.max_retries:
                    raise
                delay = max(self._backoff_delay(attempts - 1), self._retry_after(exc))
                logger.warning(f"[{request_id}] attempt {attempts} failed ({type(exc).__name__}); "
                               f"retrying in {delay:.2f}s")
                time.sleep(delay)

    # -- main API -----------------------------------------------------------
    def chat(self, messages: list, max_tokens: int = 500,
             temperature: Optional[float] = 0.0, **kwargs) -> str:
        if self.config.budget_usd is not None and self.spent_usd >= self.config.budget_usd:
            raise BudgetExceededError(f"spent ${self.spent_usd:.6f} of ${self.config.budget_usd:.6f} budget")

        request_id = uuid.uuid4().hex[:8]
        params = {"messages": messages, self.config.max_tokens_param: max_tokens, **kwargs}
        if temperature is not None:   # some reasoning models reject temperature
            params["temperature"] = temperature
        if self.config.reasoning is not None:   # OpenRouter's reasoning switch travels in the request body
            params["extra_body"] = {**params.get("extra_body", {}), "reasoning": self.config.reasoning}

        start = time.perf_counter()
        model_used, status = self.config.model, "ok"
        with self._slots:             # client-side concurrency limit
            try:
                resp, attempts = self._create_with_retries(params, request_id)
            except RETRYABLE_ERRORS + (CircuitOpenError,) as exc:
                if not self.config.fallback_model:
                    self._record(request_id=request_id, model=model_used, status="failed", attempts=0,
                                 input_tokens=0, output_tokens=0, finish_reason=None, cost_usd=0.0,
                                 latency_ms=int((time.perf_counter() - start) * 1000))
                    raise
                logger.warning(f"[{request_id}] primary failed ({type(exc).__name__}); "
                               f"trying fallback model {self.config.fallback_model}")
                model_used, status, attempts = self.config.fallback_model, "fallback_ok", 1
                resp = self.fallback_client.chat.completions.create(model=model_used, **params)

        choice, usage = resp.choices[0], resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        reasoning_tok = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        truncated = choice.finish_reason == "length"
        self._record(request_id=request_id, model=model_used,
                     status="truncated" if truncated else status, attempts=attempts,
                     input_tokens=in_tok, output_tokens=out_tok,
                     latency_ms=int((time.perf_counter() - start) * 1000),
                     cost_usd=round(self._cost(in_tok, out_tok), 6), finish_reason=choice.finish_reason,
                     reasoning_tokens=reasoning_tok)

        # Keep the assistant turn for follow-ups. OpenRouter: pass reasoning_details back unmodified.
        turn = {"role": "assistant", "content": choice.message.content}
        if getattr(choice.message, "reasoning_details", None):
            turn["reasoning_details"] = choice.message.reasoning_details
        self._turns.message = turn
        hint = ("; the model's reasoning used the whole budget: raise max_tokens or set reasoning={'enabled': False}"
                if reasoning_tok and not choice.message.content else "")

        # Never hand half an answer to the rest of the system.
        if truncated:
            raise TruncatedOutputError(f"[{request_id}] output cut off at max_tokens={max_tokens}{hint}")
        return choice.message.content

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


def summarize_calls(log: Optional[list] = None) -> dict:
    """Totals across recorded calls: the seed of cost and latency monitoring."""
    log = CALL_LOG if log is None else log
    if not log:
        return {"calls": 0}
    lat = sorted(r.latency_ms for r in log)
    return {
        "calls": len(log),
        "by_status": {s: sum(r.status == s for r in log) for s in sorted({r.status for r in log})},
        "retries": sum(max(r.attempts - 1, 0) for r in log),
        "input_tokens": sum(r.input_tokens for r in log),
        "output_tokens": sum(r.output_tokens for r in log),
        "total_cost_usd": round(sum(r.cost_usd for r in log), 6),
        "p50_latency_ms": lat[len(lat) // 2],
        "max_latency_ms": lat[-1],
    }
