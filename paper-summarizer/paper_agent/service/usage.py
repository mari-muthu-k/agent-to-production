"""Who spent what, a budget checked before every model call, and a span around every model and tool call
(Day 4, 5.3-5.6 and 6.2).

Library mode only: SPEND, CALLS and BUDGETS live in this process's memory. That is fine for one Colab process;
with several workers each would count separately, which is why production keeps them in the LiteLLM proxy
(Postgres + Redis). In proxy mode budget_guard and record_usage do nothing: the proxy checks the budget
before, and records the spend after, every call made with a user's virtual key.
"""
from collections import defaultdict

from langchain.agents.middleware import after_model, before_model, wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage

from paper_agent.service import config
from paper_agent.service.config import DEFAULT_BUDGET_USD
from paper_agent.service.errors import BudgetExceededError, from_proxy_error
from paper_agent.service.gateway import deployment_of
from paper_agent.service.tracing import TRACER

SPEND: dict = defaultdict(float)      # user -> USD spent today
CALLS: dict = defaultdict(int)        # user -> model calls today
BUDGETS: dict = {}                    # user -> USD per day; anyone else gets DEFAULT_BUDGET_USD


def message_cost(message) -> float:
    """USD for one model call: the token counts the provider reported, at our prices (config)."""
    usage = getattr(message, "usage_metadata", None) or {}
    return (usage.get("input_tokens", 0) * config.PRICE_IN_PER_M
            + usage.get("output_tokens", 0) * config.PRICE_OUT_PER_M) / 1e6


@after_model
def record_usage(state, runtime):
    """After every model call: add its cost to the user's spend."""
    if config.GATEWAY_MODE == "proxy":
        return None
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        SPEND[runtime.context.user_id] += message_cost(last)
        CALLS[runtime.context.user_id] += 1
    return None


def check_budget(user_id: str) -> None:
    """Raise BudgetExceededError once the user has spent their whole budget."""
    spent = SPEND[user_id]
    budget = BUDGETS.get(user_id, DEFAULT_BUDGET_USD)
    if spent >= budget:
        raise BudgetExceededError(user_id, spent, budget)


@before_model
def budget_guard(state, runtime):
    """Before EVERY model call, not once per question: one question can take several calls."""
    if config.GATEWAY_MODE == "proxy":
        return None
    check_budget(runtime.context.user_id)
    return None


@wrap_model_call
def trace_model_call(request, handler):
    """An llm.call span with the deployment that answered, the tokens and the cost. In proxy mode it also
    turns the proxy's budget and rate-limit errors into ours (errors.from_proxy_error)."""
    with TRACER.start_as_current_span("llm.call") as span:
        try:
            response = handler(request)
        except Exception as exc:
            mapped = from_proxy_error(exc, getattr(request.runtime.context, "user_id", "?"))
            if mapped is not None:
                raise mapped from exc
            raise
        message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        usage = (message.usage_metadata if message else None) or {}
        span.set_attribute("llm.deployment", deployment_of(message) or "unknown")
        span.set_attribute("llm.input_tokens", usage.get("input_tokens", 0))
        span.set_attribute("llm.output_tokens", usage.get("output_tokens", 0))
        span.set_attribute("llm.cost_usd", round(message_cost(message), 8))
        return response


@wrap_tool_call
def trace_tool_call(request, handler):
    """A tool.<name> span around every tool call."""
    with TRACER.start_as_current_span(f"tool.{request.tool_call['name']}"):
        return handler(request)


def usage_of(user_id: str) -> dict:
    """GET /me/usage in library mode."""
    budget = BUDGETS.get(user_id, DEFAULT_BUDGET_USD)
    return {"user": user_id, "llm_calls": CALLS[user_id], "spent_usd": round(SPEND[user_id], 6),
            "budget_usd": round(budget, 6), "remaining_usd": round(budget - SPEND[user_id], 6)}
