"""The chat model for the agent (Day 3, 3.1): ChatOpenAI pointed at the same endpoint as Days 1-2.

LLM_BASE_URL alone chooses the provider (mock-llm offline, a gateway or provider online): no code
changes. Keys come from the environment (Colab Secrets or .env), never from code.

Gemini's OpenAI-compatible endpoint attaches a `thought_signature` to tool calls (in each tool call's
`extra_content`) and rejects the next request with a 400 unless it comes back unchanged. langchain-openai
1.6.7 drops that field, so ToolCallSignatures keeps it on the AIMessage and puts it back, matched by position
(the n-th tool call of the n-th assistant turn), so a rewritten tool-call id can't break the match. If a Gemini
tool call still has no signature, it sends Google's documented placeholder "skip_thought_signature_validator"
(a last resort: the model loses that turn's reasoning) and warns once. Other providers are unaffected.
"""
import os
import warnings

from langchain_openai import ChatOpenAI

MAX_TOKENS = 1000
EXTRA = "tool_call_extra_content"         # AIMessage.additional_kwargs key: [extra_content per tool call, in order]
SKIP_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


class ToolCallSignatures(ChatOpenAI):
    """ChatOpenAI that round-trips each tool call's provider extras (Gemini's thought signatures)."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw.get("choices") or [], strict=False):
            extras = [c.get("extra_content") for c in (choice.get("message") or {}).get("tool_calls") or []]
            if any(extras):
                generation.message.additional_kwargs[EXTRA] = extras
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        sent = [m for m in payload.get("messages") or [] if m.get("role") == "assistant" and m.get("tool_calls")]
        ours = [m for m in self._convert_input(input_).to_messages()
                if getattr(m, "tool_calls", None) or getattr(m, "invalid_tool_calls", None)]
        gemini = "gemini" in (self.model_name or "").lower()
        for message, original in zip(sent, ours, strict=False):
            extras = original.additional_kwargs.get(EXTRA) or []
            for i, call in enumerate(message["tool_calls"]):
                if i < len(extras) and extras[i]:
                    call["extra_content"] = extras[i]
                elif gemini:
                    call["extra_content"] = SKIP_SIGNATURE
                    warnings.warn("a Gemini tool call came back without its thought signature; sent Google's "
                                  "placeholder instead (the model loses that turn's reasoning)", stacklevel=2)
        return payload


def make_chat_model(cache=None, **overrides) -> ChatOpenAI:
    """ChatOpenAI (keeping Gemini's tool-call signatures) from LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
    (the first model, if it is a list).
    cache: e.g. InMemoryCache() for an exact-match response cache on this model only (6.2)."""
    settings = dict(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL"].split(",")[0].strip(),
        temperature=0,
        timeout=30,
        max_tokens=MAX_TOKENS,
        # One retry layer (the Day 1 rule): the client retries twice; no ModelRetryMiddleware on top.
        # Day 4 moves retries to the LiteLLM gateway and sets this to 0.
        max_retries=2,
        cache=cache,
    )
    if os.environ.get("LLM_MAX_TOKENS_PARAM") == "max_tokens":   # ChatOpenAI sends max_completion_tokens;
        settings["max_tokens"] = None                            # some endpoints only accept max_tokens
        settings["extra_body"] = {"max_tokens": MAX_TOKENS}
    settings.update(overrides)
    return ToolCallSignatures(**settings)
