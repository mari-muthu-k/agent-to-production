"""OpenRouter specifics: reasoning switch, reasoning_details round-trips, budgets, rate-limit messages."""
import json

import openai
import pytest

from paper_agent.llm_client import CALL_LOG, LLMClient, LLMConfig, TruncatedOutputError
from paper_agent.openrouter import (
    REASONING_OFF,
    REASONING_ON,
    assistant_turn,
    is_openrouter,
    make_client,
    rate_limit_message,
    reasoning_body,
    reasoning_tokens,
)
from paper_agent.schemas import Answer
from paper_agent.testing import FakeLLM, fake_response

STRAWBERRY = [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]


def test_make_client_adds_attribution_only_for_openrouter():
    orc = make_client("k", "https://openrouter.ai/api/v1")
    assert orc.default_headers["X-Title"] == "Paper Summarizer Workshop" and orc.max_retries == 0
    other = make_client("k", "http://mock-llm:8000/v1")
    assert "X-Title" not in other.default_headers
    assert is_openrouter("https://openrouter.ai/api/v1") and not is_openrouter("http://mock-llm:8000/v1")


def test_reasoning_body_merges():
    assert reasoning_body(None, {"a": 1}) == {"a": 1}
    assert reasoning_body(REASONING_OFF, {"a": 1}) == {"a": 1, "reasoning": {"enabled": False}}


def test_llm_client_sends_reasoning_in_extra_body():
    seen = {}

    class Spy(FakeLLM):
        def _create(self, **kwargs):
            seen.update(kwargs)
            return super()._create(**kwargs)

    LLMClient(LLMConfig(model="x", reasoning={"effort": "low"}), client=Spy([fake_response("ok")])).chat(
        STRAWBERRY, extra_body={"provider": {"sort": "price"}})
    assert seen["extra_body"] == {"provider": {"sort": "price"}, "reasoning": {"effort": "low"}}
    seen.clear()
    LLMClient(LLMConfig(model="x"), client=Spy([fake_response("ok")])).chat(STRAWBERRY)
    assert "extra_body" not in seen                                  # None = the model's default


def test_mock_reasoning_changes_the_answer_and_returns_details(mock):
    c = mock.client()
    plain = c.chat.completions.create(model="m", max_tokens=200, messages=STRAWBERRY)
    thinking = c.chat.completions.create(model="m", max_tokens=200, messages=STRAWBERRY,
                                         extra_body={"reasoning": REASONING_ON})
    assert plain.choices[0].message.content == "2" and reasoning_tokens(plain.usage) == 0
    m = thinking.choices[0].message
    assert m.content == "3" and "s-t-r-a-w-b-e-r-r-y" in m.reasoning
    assert m.reasoning_details[0]["type"] == "reasoning.text" and reasoning_tokens(thinking.usage) > 0
    assert thinking.usage.completion_tokens == reasoning_tokens(thinking.usage) + 1


def test_reasoning_details_round_trip(mock):
    c = mock.client()
    first = c.chat.completions.create(model="m", max_tokens=200, messages=STRAWBERRY,
                                      extra_body={"reasoning": REASONING_ON})
    turn = assistant_turn(first.choices[0].message)
    assert turn["reasoning_details"] and turn["content"] == "3"
    sure = {"role": "user", "content": "Are you sure? Think carefully."}
    follow = c.chat.completions.create(model="m", max_tokens=200, extra_body={"reasoning": REASONING_ON},
                                       messages=STRAWBERRY + [turn, sure])
    assert "re-checked" in follow.choices[0].message.content
    without = c.chat.completions.create(model="m", max_tokens=200, messages=STRAWBERRY + [
        {"role": "assistant", "content": "3"}, {"role": "user", "content": "Are you sure? Think carefully."}])
    assert "recount" in without.choices[0].message.content


def test_reasoning_can_eat_the_whole_budget(mock):
    llm = LLMClient(LLMConfig(model="m", reasoning=REASONING_ON), client=mock.client())
    with pytest.raises(TruncatedOutputError, match="reasoning used the whole budget"):
        llm.chat(STRAWBERRY, max_tokens=5)
    assert CALL_LOG[-1].reasoning_tokens == 5 and CALL_LOG[-1].status == "truncated"
    assert LLMClient(LLMConfig(model="m", reasoning=REASONING_OFF), client=mock.client()).chat(STRAWBERRY,
                                                                                              max_tokens=5) == "2"


def test_last_message_keeps_reasoning_details(mock):
    llm = LLMClient(LLMConfig(model="m", reasoning=REASONING_ON), client=mock.client())
    assert llm.chat(STRAWBERRY, max_tokens=200) == "3"
    assert llm.last_message["reasoning_details"][0]["text"].startswith("Spell it out")


def test_validation_repair_sends_reasoning_details_back():
    sent = []

    class Spy(FakeLLM):
        def _create(self, **kwargs):
            sent.append(kwargs["messages"])
            return super()._create(**kwargs)

    bad = fake_response('{"answer": "x"}')
    bad.choices[0].message.reasoning_details = [{"type": "reasoning.text", "text": "thinking"}]
    good = fake_response(json.dumps({"answer": "3", "citations": ["results"], "confidence": "high"}))
    LLMClient(LLMConfig(model="m"), client=Spy([bad, good])).chat_structured(STRAWBERRY, Answer)
    assistant = sent[1][-2]
    assert assistant["role"] == "assistant" and assistant["reasoning_details"][0]["text"] == "thinking"


def test_rate_limit_message_reads_openrouter_headers(mock):
    with pytest.raises(openai.RateLimitError) as e:
        mock.client().chat.completions.create(model="m", max_tokens=5, messages=STRAWBERRY,
                                              extra_headers={"X-Mock-Status": "429", "X-Mock-Retry-After": "7"})
    assert e.value.response.headers["x-ratelimit-remaining"] == "0"
    assert "limit 20, remaining 0" in rate_limit_message(e.value) and "50/day" in rate_limit_message(e.value)


def test_bad_reasoning_type_is_400(mock):
    with pytest.raises(openai.BadRequestError):
        mock.client().chat.completions.create(model="m", max_tokens=5, messages=STRAWBERRY,
                                              extra_body={"reasoning": "yes"})
