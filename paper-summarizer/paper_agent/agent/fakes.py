"""An offline chat model for the Day 3 TODO tests: no network, no key, instant, deterministic.

langchain_core's GenericFakeChatModel has no bind_tools(), which create_agent needs, so this is a
small fake of our own. With `script`, it returns those AIMessages in order (tool calls or answers).
Without one, it picks a tool from the tool descriptions only, the same rule mock-llm uses
(paper_agent/agent/policy.py): that is what makes the 3.4 docstring test meaningful offline.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from paper_agent.agent.policy import pick_tool


class FakeToolModel(BaseChatModel):
    script: list = Field(default_factory=list)       # AIMessages to return, in order
    tools: list = Field(default_factory=list)        # what bind_tools() received, as OpenAI function dicts
    requests: list = Field(default_factory=list)     # the messages sent on every call

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):
        # a shallow copy: script and requests stay shared with the original, as with a real bound model
        return self.model_copy(update={"tools": [convert_to_openai_tool(t)["function"] for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.requests.append(list(messages))
        if self.script:
            reply = self.script.pop(0)
        elif self.tools and not isinstance(messages[-1], ToolMessage):
            question = next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            name, args = pick_tool(question, self.tools)
            reply = AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{len(self.requests)}"}])
        else:
            reply = AIMessage("Done.")
        n_in = sum(len(str(m.content)) for m in messages) // 4 + 50 * len(self.tools)
        reply = reply.model_copy(update={"usage_metadata": {"input_tokens": n_in, "output_tokens": 20,
                                                            "total_tokens": n_in + 20}})
        return ChatResult(generations=[ChatGeneration(message=reply)])
