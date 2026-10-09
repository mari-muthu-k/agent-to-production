"""The Day 3 agent as something a service can call (Day 4, 4.0-4.1): build_agent() and ask_paper().

Same tools, same untrusted-text guard and same citation check as Day 3. New: the model sits behind the gateway,
the prompt is versioned (prompts/agent_v1.txt, agent_v2.txt), every model and tool call is traced and costed,
and ask_paper() returns what a service needs to report: citations, calls, tokens, cost, who answered.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage

from paper_agent.agent import tools as day3_tools
from paper_agent.agent.guards import CHUNK_ID, CITATION_FAILED, cited_ids, retrieved_ids, untrusted_content_guard
from paper_agent.agent.papers import Context
from paper_agent.agent.tools import chunk_tags, get_section
from paper_agent.agent.trace import this_run
from paper_agent.prompts import load_prompt
from paper_agent.rag import pdf_rag as day2
from paper_agent.service import config, store
from paper_agent.service.gateway import CURRENT_KEY, deployment_of, make_chat_model
from paper_agent.service.tracing import TRACER
from paper_agent.service.usage import message_cost, record_usage, trace_model_call, trace_tool_call

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")
EXPLAIN_Q = "Give me an overview of this paper: the main result and its limitations."


@dataclass
class Ctx(Context):
    """Day 3's Context (user_id, paper_id), plus the caller's virtual key in proxy mode."""
    api_key: Optional[str] = field(default=None, repr=False)


@tool("search_paper", description=day3_tools.search_paper.description)
def search_paper(query: str, runtime: ToolRuntime[Ctx], k: int = 4) -> str:
    """Day 3's search_paper, retrieving through store.search() (traced; 6.3 slows it down)."""
    k = min(max(int(k), 1), 8)
    CURRENT_KEY.set(getattr(runtime.context, "api_key", None))          # proxy mode: embed with the caller's key
    hits = store.search(query, paper_id=runtime.context.paper_id, k=k)
    return chunk_tags(hits) or "No passages found in this paper."


TOOLS = [search_paper, get_section]
_USER_MODELS: dict = {}       # proxy mode: one client per virtual key (a cache: nothing shared lives here)


@wrap_model_call
def user_key_model(request, handler):
    """Proxy mode: send each model call with the caller's own virtual key, so the proxy applies their budget
    and rate limit."""
    key = getattr(request.runtime.context, "api_key", None)
    if not key:
        return handler(request)
    if key not in _USER_MODELS:
        _USER_MODELS[key] = make_chat_model(user_key=key)
    return handler(request.override(model=_USER_MODELS[key]))


def build_agent(prompt_version: str = PROMPT_VERSION, extra_middleware=(), model=None, router=None):
    """The Day 3 agent with the Day 4 plumbing: a model-call limit, usage recording, spans around every
    model and tool call, and Day 3's untrusted-text guard. extra_middleware: e.g. [budget_guard] (5.4)."""
    middleware = [ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"), record_usage, trace_model_call,
                  trace_tool_call, untrusted_content_guard, *extra_middleware]
    if config.GATEWAY_MODE == "proxy":
        middleware.append(user_key_model)
    agent = create_agent(model or make_chat_model(router), tools=TOOLS, middleware=middleware, context_schema=Ctx,
                         system_prompt=load_prompt("agent", prompt_version))
    agent.prompt_version = prompt_version
    return agent


def check_answer(messages: list) -> tuple:
    """Day 3's citation check, in its own span: (answer to show, cited ids, valid?)."""
    with TRACER.start_as_current_span("citation_check") as span:
        answer = messages[-1].text if messages and isinstance(messages[-1], AIMessage) else ""
        cited, retrieved = cited_ids(answer), retrieved_ids(messages)
        try:
            day2.check_citations(day2.GroundedAnswer(answer=answer, found=bool(cited), citations=cited),
                                 [{"id": i} for i in retrieved])
            valid = True
        except day2.CitationError:
            valid, answer = False, CITATION_FAILED.format(ids=sorted(set(cited) - retrieved))
        span.set_attribute("citation.valid", valid)
        return answer, cited, valid


def summarize_run(messages: list, started: float) -> dict:
    """What one question cost: model calls, tokens, USD and which deployments answered."""
    calls = [m for m in this_run(messages) if isinstance(m, AIMessage) and m.usage_metadata]
    return {"llm_calls": len(calls),
            "input_tokens": sum(m.usage_metadata.get("input_tokens", 0) for m in calls),
            "output_tokens": sum(m.usage_metadata.get("output_tokens", 0) for m in calls),
            "cost_usd": round(sum(message_cost(m) for m in calls), 6),
            "answered_by": list(dict.fromkeys(deployment_of(m) or "unknown" for m in calls)),
            "latency_ms": round((time.perf_counter() - started) * 1000)}


def ask_paper(paper_id: str, question: str, user_id: str, agent=None, api_key: Optional[str] = None) -> dict:
    """One question through the agent: the checked answer, its citations and the usage of this run."""
    agent = agent or build_agent()
    started = time.perf_counter()
    with TRACER.start_as_current_span("agent.run") as span:
        span.set_attribute("agent.prompt_version", getattr(agent, "prompt_version", "?"))
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, {"recursion_limit": 50},
                              context=Ctx(user_id, paper_id, api_key))
    messages = result["messages"]
    answer, cited, valid = check_answer(messages)
    return {"answer": answer, "citations": cited, "citations_valid": valid,
            "retrieved": sorted(retrieved_ids(messages)), **summarize_run(messages, started)}


def explain_events(paper_id: str, question: str, user_id: str, agent=None, api_key: Optional[str] = None):
    """The same run, as events while it happens (4.6): progress..., answer, done."""
    agent = agent or build_agent()
    started, seen = time.perf_counter(), []
    yield "progress", {"step": "started"}
    with TRACER.start_as_current_span("agent.run"):
        for update in agent.stream({"messages": [{"role": "user", "content": question}]}, {"recursion_limit": 50},
                                   context=Ctx(user_id, paper_id, api_key), stream_mode="updates"):
            for node, change in update.items():
                messages = (change or {}).get("messages", []) if isinstance(change, dict) else []
                seen.extend(messages)
                for m in messages:
                    if node == "model" and isinstance(m, AIMessage) and m.tool_calls:
                        yield "progress", {"step": "tool", "tools": [c["name"] for c in m.tool_calls]}
                    elif node == "tools" and isinstance(m, ToolMessage):
                        yield "progress", {"step": "retrieved", "chunks": CHUNK_ID.findall(str(m.content))}
    answer, cited, valid = check_answer(seen)
    yield "answer", {"answer": answer, "citations": cited, "citations_valid": valid}
    yield "done", summarize_run(seen, started)
