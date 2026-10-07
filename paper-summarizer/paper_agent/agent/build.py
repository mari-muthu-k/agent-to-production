"""build_agent(): the agent the notebook and the tests use (Day 3, 4.1 and 5.8)."""
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

from paper_agent.agent.guards import citation_check, tool_error_message, untrusted_content_guard
from paper_agent.agent.memory import STORE, reader_profile
from paper_agent.agent.model import make_chat_model
from paper_agent.agent.papers import Context
from paper_agent.agent.prompts import AGENT_SYSTEM_V1
from paper_agent.agent.tools import TOOLS


def guard_stack() -> list:
    """The five guard layers, in order. Order matters: before_* hooks run top to bottom, after_* hooks
    bottom to top, and wrap_* hooks nest with the first one outermost, as in a web framework. So the
    limits are checked before anything else runs, PII is redacted before the model reads a tool result,
    every tool result is wrapped before the model sees it, ToolErrorMiddleware sits inside the injection
    guard (an error message is wrapped too), and the citation check sees the final answer last."""
    return [
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),          # 1. limits: normal questions
        ToolCallLimitMiddleware(run_limit=8),                                #    need 2-4 model calls
        PIIMiddleware("email", strategy="redact", apply_to_tool_results=True),   # 2. personal data
        untrusted_content_guard,                                             # 3. injection guard (TODO 2)
        HumanInTheLoopMiddleware(interrupt_on={"save_note": True}),          # 4. a person approves writes
        citation_check,                                                      # 5. output check
        ToolErrorMiddleware(on_error=tool_error_message),                    # safety net: crash -> message
    ]


def build_agent(guarded: bool = False, tools=None, middleware=None, model=None, checkpointer=None, store=None,
                system_prompt: str = AGENT_SYSTEM_V1):
    """create_agent with our tools, prompt, memory and (guarded=True) the guard stack. Pass `middleware`
    to choose the layers yourself; the reader-profile middleware is always added last."""
    stack = list(middleware) if middleware is not None else (guard_stack() if guarded else [])
    if reader_profile not in stack:
        stack.append(reader_profile)
    return create_agent(
        model or make_chat_model(),
        tools=list(tools or TOOLS),
        system_prompt=system_prompt,
        middleware=stack,
        checkpointer=checkpointer or InMemorySaver(),        # short-term memory: one conversation per thread_id
        store=STORE if store is None else store,             # long-term memory: notes per user_id
        context_schema=Context,                              # which paper and which reader, per run
    )


def run_config(thread_id: str, recursion_limit: int = 100) -> dict:
    """recursion_limit counts graph steps, and every middleware hook is a step too: it is a backstop, not a
    model-call limit (ModelCallLimitMiddleware is). 4.5 sets 25 on an agent without guards."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
