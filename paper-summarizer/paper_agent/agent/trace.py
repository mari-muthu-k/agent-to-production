"""Reading what the agent did (Day 3, 4.1-4.2): a streamed trace, then one table row per model call.

Costs use the Day 1 price config. Free models cost $0 per token; the prices show the maths you would
pay on a paid model, and why an agent answer costs several times a single RAG answer.
"""
import uuid
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

PRICE_IN_PER_1M = 0.50     # Day 1 price config: USD per 1M input tokens
PRICE_OUT_PER_1M = 2.00    # USD per 1M output tokens


def this_run(messages: list) -> list:
    """The messages after the last user message: what the agent did for the current question."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human + 1:]


def call_cost(message) -> float:
    """USD for one model call, from the token counts the provider reported (0 for a cache hit)."""
    usage = getattr(message, "usage_metadata", None) or {}
    if usage.get("total_cost") == 0:          # LangChain's mark on a reply served from its cache
        return 0.0
    return (usage.get("input_tokens", 0) * PRICE_IN_PER_1M + usage.get("output_tokens", 0) * PRICE_OUT_PER_1M) / 1e6


def describe_call(call: dict, width: int = 70) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
    text = f"{call['name']}({args})"
    return text if len(text) <= width else text[:width - 1] + "…)"


def describe_result(message: ToolMessage) -> str:
    text = str(message.content)
    ids = [part.split('"')[1] for part in text.split('<chunk id=')[1:]]
    summary = f"{len(ids)} chunk(s): {', '.join(ids)}" if ids else " ".join(text.split())[:90]
    if "GUARD NOTE" in text:
        summary += "  + 🛡 GUARD NOTE"
    return f"{message.name} → {summary}"


@dataclass
class Trace:
    messages: list = field(default_factory=list)
    interrupts: list = field(default_factory=list)
    error: Optional[Exception] = None

    @property
    def answer(self) -> str:
        last = this_run(self.messages)[-1:] if not self.interrupts and not self.error else []
        return last[0].text if last and isinstance(last[0], AIMessage) else ""

    @property
    def calls(self) -> list:
        return [m for m in this_run(self.messages) if isinstance(m, AIMessage) and m.usage_metadata]

    def cost_line(self) -> str:
        n_in = sum(m.usage_metadata.get("input_tokens", 0) for m in self.calls)
        n_out = sum(m.usage_metadata.get("output_tokens", 0) for m in self.calls)
        return (f"💰 {len(self.calls)} model call(s), {n_in:,} input + {n_out:,} output tokens = "
                f"${sum(call_cost(m) for m in self.calls):.4f} at Day 1 prices")


def stream_trace(agent, question, config: Optional[dict] = None, context=None, show: bool = True) -> Trace:
    """Run the agent and print each step as it happens. `question` is the user's text, or a Command
    (resuming after an approval). Returns a Trace; a GraphRecursionError is caught and kept in .error."""
    config = config or {"configurable": {"thread_id": uuid.uuid4().hex}}
    payload = {"messages": [{"role": "user", "content": question}]} if isinstance(question, str) else question
    say = print if show else (lambda *a, **k: None)
    trace, seen, n_model = Trace(), [], 0
    if isinstance(question, str):
        say(f"🧑 {question}")
    try:
        for update in agent.stream(payload, config, context=context, stream_mode="updates"):
            for node, change in update.items():
                if node == "__interrupt__":
                    trace.interrupts.extend(change)
                    for item in change:
                        for action in item.value.get("action_requests", []):
                            say(f"   ⏸  paused for approval: {describe_call(action)}")
                    continue
                messages = (change or {}).get("messages", []) if isinstance(change, dict) else []
                seen.extend(messages)
                for m in messages:
                    if node == "model" and isinstance(m, AIMessage):
                        n_model += 1
                        what = ", ".join(describe_call(c) for c in m.tool_calls) or "answer: " + m.text[:400]
                        say(f"#{n_model:<2} 🧠 model → {what}")
                    elif node == "tools" and isinstance(m, ToolMessage):
                        say(f"    🔧 {describe_result(m)}")
                if node not in ("model", "tools") and messages:
                    say(f"    🛡  {node.split('.')[0]}: {middleware_effect(messages)}")
    except GraphRecursionError as e:
        trace.error = e
        say(f"💥 GraphRecursionError: {str(e).splitlines()[0]}")
    state = agent.get_state(config) if getattr(agent, "checkpointer", None) else None
    trace.messages = list(state.values.get("messages", [])) if state else seen
    if show and not trace.interrupts:
        say(trace.cost_line())
    return trace


def middleware_effect(messages: list) -> str:
    text = " ".join(str(m.content) for m in messages)
    if "[REDACTED_" in text:
        return "redacted personal data in the conversation"
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return f"{last.name}: {' '.join(str(last.content).split())[:120]}"
    return " ".join(str(last.content).split())[:200]


HOOKS = [("before_agent", "once, before the agent starts"), ("before_model", "before every model call"),
         ("wrap_model_call", "around every model call"), ("after_model", "after every model call"),
         ("wrap_tool_call", "around every tool call"), ("after_agent", "once, after the agent finishes")]


def hook_map(middleware: list) -> None:
    """Print which middleware runs at which hook (5.1): the hooks each one overrides."""
    from langchain.agents.middleware import AgentMiddleware
    table = {hook: [m.name for m in middleware if getattr(type(m), hook, None) is not getattr(AgentMiddleware, hook)]
             for hook, _ in HOOKS}
    for hook, when in HOOKS:
        print(f"{hook:16} {when:32} {', '.join(table[hook]) or '-'}")


def trace_table(messages: list, show: bool = True) -> list:
    """One row per model call in the last run: what it did, tokens in (of which cached), tokens out, cost."""
    rows = []
    for m in this_run(messages):
        if not isinstance(m, AIMessage) or not m.usage_metadata:
            continue
        details = m.usage_metadata.get("input_token_details") or {}
        rows.append({"call": len(rows) + 1,
                     "step": ", ".join(describe_call(c, 44) for c in m.tool_calls) or "final answer",
                     "input": m.usage_metadata.get("input_tokens", 0), "cached": details.get("cache_read") or 0,
                     "output": m.usage_metadata.get("output_tokens", 0), "cost": call_cost(m)})
    if show:
        print(f"{'#':>2}  {'step':46} {'input':>7} {'cached':>7} {'output':>7} {'cost $':>9}")
        for r in rows:
            print(f"{r['call']:>2}  {r['step'][:46]:46} {r['input']:>7,} {r['cached']:>7,} {r['output']:>7,} "
                  f"{r['cost']:>9.6f}")
        print(f"    {'total':46} {sum(r['input'] for r in rows):>7,} {sum(r['cached'] for r in rows):>7,} "
              f"{sum(r['output'] for r in rows):>7,} {sum(r['cost'] for r in rows):>9.6f}")
    return rows
