"""The two errors a user can cause by using the service too much, and how the proxy's errors map onto them.

Library mode raises these from our own code (usage.check_budget, the rate limiter in app.py). Proxy mode gets
the LiteLLM proxy's HTTP errors back through the OpenAI client; from_proxy_error() turns them into the same
two exceptions, so the user sees the same friendly 429 body in both modes.
"""
import re
from typing import Optional


class BudgetExceededError(Exception):
    def __init__(self, user_id: str, spent: float, budget: float):
        self.user_id, self.spent, self.budget = user_id, spent, budget
        super().__init__(f"{user_id} has spent ${spent:.6f} of a ${budget:.6f} budget")


class RateLimitedError(Exception):
    def __init__(self, user_id: str, retry_after: Optional[int] = None, limit: Optional[int] = None):
        self.user_id, self.retry_after, self.limit = user_id, retry_after, limit
        super().__init__(f"{user_id} is over {limit or 'the'} requests per minute")


def _retry_after(exc) -> Optional[int]:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    value = headers.get("retry-after")
    try:
        return max(1, round(float(value))) if value is not None else None
    except ValueError:
        return None


def from_proxy_error(exc: BaseException, user_id: str = "?") -> Optional[Exception]:
    """The proxy's budget or rate-limit error as ours; None for anything else.

    Measured on LiteLLM proxy 1.104.2 (docs/instructor_notes_day4.md):
      over budget   HTTP 422, error.type "budget_exceeded", message "Budget has been exceeded! Key=bob (sk-...)
                    Current cost: 0.000147, Max budget: 0.0001"; no Retry-After
      rate limited  HTTP 429, error.type "throttling_error", message "Rate limit exceeded for api_key: ...
                    Limit type: requests. Current limit: 2, Remaining: 0. ...", header Retry-After: 60
    """
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    kind = str((error or {}).get("type") or "")
    text = str((error or {}).get("message") or exc)
    if kind == "budget_exceeded" or re.search(r"budget has been exceeded|exceededbudget|over budget", text, re.I):
        spent = re.search(r"(?:current cost|spend)\s*[:=]\s*([\d.e-]+)", text, re.I)
        budget = re.search(r"max(?:imum)?[ _]budget\s*[:=]\s*([\d.e-]+)", text, re.I)
        return BudgetExceededError(user_id, float(spent.group(1)) if spent else 0.0,
                                   float(budget.group(1)) if budget else 0.0)
    if status == 429:
        return RateLimitedError(user_id, retry_after=_retry_after(exc))
    return None
