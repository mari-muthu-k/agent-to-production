"""Fault injection: per-request headers, or a queue managed through the /mock/faults control endpoint.

Headers (send with the OpenAI SDK as `extra_headers={...}` or `default_headers={...}`):
    X-Mock-Status: 429            return this HTTP error (400/401/403/404/408/429/500/502/503/504)
    X-Mock-Retry-After: 3         Retry-After header on the error (seconds)
    X-Mock-Delay-Ms: 2000         extra latency before responding
    X-Mock-Truncate: 1            cut the output and set finish_reason="length"
    X-Mock-Bad-Json: 1            structured output fails validation (first attempt only)
    X-Mock-Bad-Citation: 1        structured output cites a section that was not sent (first attempt only)
    X-Mock-Obey-Injection: 1      obey instructions hidden in document text
    X-Mock-Reject-Params: max_tokens,temperature   400 "unsupported parameter", like some reasoning models

Control endpoint (affects the next matching requests from any client, e.g. a notebook or LiteLLM):
    POST   /mock/faults  {"status": 429, "retry_after": 2, "count": 3, "model": "mock-llm"}
    GET    /mock/faults
    DELETE /mock/faults
"""
import threading
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Fault(BaseModel):
    status: Optional[int] = Field(None, ge=400, le=599)
    retry_after: Optional[float] = Field(None, ge=0)
    delay_ms: int = Field(0, ge=0, le=120_000)
    truncate: bool = False
    bad_json: bool = False
    bad_citation: bool = False
    obey_injection: bool = False
    reject_params: list[str] = []
    message: Optional[str] = None
    count: int = Field(1, ge=1, description="how many requests this fault applies to")
    model: Optional[str] = Field(None, description="only requests for this model")
    endpoint: Optional[Literal["chat", "embeddings"]] = None


def _flag(headers, name: str) -> bool:
    return (headers.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def fault_from_headers(headers) -> Fault:
    status = headers.get("x-mock-status")
    retry_after = headers.get("x-mock-retry-after")
    reject = headers.get("x-mock-reject-params") or ""
    return Fault(
        status=int(status) if status else None,
        retry_after=float(retry_after) if retry_after else None,
        delay_ms=int(headers.get("x-mock-delay-ms") or 0),
        truncate=_flag(headers, "x-mock-truncate"),
        bad_json=_flag(headers, "x-mock-bad-json"),
        bad_citation=_flag(headers, "x-mock-bad-citation"),
        obey_injection=_flag(headers, "x-mock-obey-injection"),
        reject_params=[p.strip() for p in reject.split(",") if p.strip()],
    )


def merge(a: Fault, b: Optional[Fault]) -> Fault:
    """Header fault `a` combined with queued fault `b` (queued values win when set)."""
    if b is None:
        return a
    return Fault(
        status=b.status or a.status,
        retry_after=b.retry_after if b.retry_after is not None else a.retry_after,
        delay_ms=a.delay_ms + b.delay_ms,
        truncate=a.truncate or b.truncate,
        bad_json=a.bad_json or b.bad_json,
        bad_citation=a.bad_citation or b.bad_citation,
        obey_injection=a.obey_injection or b.obey_injection,
        reject_params=a.reject_params + b.reject_params,
        message=b.message or a.message,
    )


class FaultQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._faults: list = []

    def add(self, fault: Fault) -> None:
        with self._lock:
            self._faults.append(fault)

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()

    def list(self) -> list:
        with self._lock:
            return [f.model_dump() for f in self._faults]

    def take(self, endpoint: str, model: Optional[str]) -> Optional[Fault]:
        """Consume one use of the first fault matching this request."""
        with self._lock:
            for i, f in enumerate(self._faults):
                if (f.endpoint in (None, endpoint)) and (f.model in (None, model)):
                    if f.count <= 1:
                        self._faults.pop(i)
                    else:
                        self._faults[i] = f.model_copy(update={"count": f.count - 1})
                    return f
        return None
