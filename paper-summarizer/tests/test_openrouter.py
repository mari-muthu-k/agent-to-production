"""OpenRouter over plain HTTP: reasoning switch, reasoning_details round-trips, budgets, model switching."""
import json

import pytest

from paper_agent.llm_client import (
    CALL_LOG,
    BadRequestError,
    LLMClient,
    LLMConfig,
    OpenRouterClient,
    RateLimitError,
    TruncatedOutputError,
)
from paper_agent.openrouter import (
    REASONING_OFF,
    REASONING_ON,
    assistant_turn,
    current_models,
    free_models,
    is_openrouter,
    list_models,
    rate_limit_message,
    reasoning_tokens,
    stream_chat,
    text_of,
    use_model,
)
from paper_agent.schemas import Answer
from paper_agent.testing import FakeLLM, fake_response

STRAWBERRY = [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]


def test_client_reads_key_and_url_from_env_at_call_time(monkeypatch):
    client = OpenRouterClient()
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-one")
    assert client.base_url == "https://openrouter.ai/api/v1" and client.headers()["Authorization"] == "Bearer sk-or-one"
    monkeypatch.setenv("LLM_API_KEY", "sk-or-two")                   # switched mid-session
    assert client.headers()["Authorization"] == "Bearer sk-or-two"
    assert client.headers()["X-Title"] == "Paper Summarizer Workshop"
    assert is_openrouter() and not is_openrouter("http://mock-llm:8000/v1")


def test_use_model_switches_everything_that_reads_the_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "a:free")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "z:free")
    assert use_model("b:free, c:free") == ["b:free", "c:free", "z:free"]
    assert current_models() == ["b:free", "c:free", "z:free"]


def test_llm_client_sends_reasoning_in_the_body():
    seen = {}

    class Spy(FakeLLM):
        def chat(self, **body):
            seen.update(body)
            return super().chat(**body)

    LLMClient(LLMConfig(model="x", reasoning={"effort": "low"}), client=Spy([fake_response("ok")])).chat(STRAWBERRY)
    assert seen["reasoning"] == {"effort": "low"} and seen["model"] == "x"
    seen.clear()
    LLMClient(LLMConfig(model="x"), client=Spy([fake_response("ok")])).chat(STRAWBERRY)
    assert "reasoning" not in seen                                     # None = the model's default


def test_mock_reasoning_changes_the_answer_and_returns_details(mock):
    c = mock.client()
    plain = c.chat(model="m", max_tokens=200, messages=STRAWBERRY)
    thinking = c.chat(model="m", max_tokens=200, messages=STRAWBERRY, reasoning=REASONING_ON)
    assert text_of(plain) == "2" and reasoning_tokens(plain.get("usage")) == 0
    m = thinking["choices"][0]["message"]
    assert m["content"] == "3" and "s-t-r-a-w-b-e-r-r-y" in m["reasoning"]
    assert m["reasoning_details"][0]["type"] == "reasoning.text"
    assert thinking["usage"]["completion_tokens"] == reasoning_tokens(thinking["usage"]) + 1


def test_reasoning_details_round_trip(mock):
    c = mock.client()
    first = c.chat(model="m", max_tokens=200, messages=STRAWBERRY, reasoning=REASONING_ON)
    turn = assistant_turn(first["choices"][0]["message"])
    assert turn["reasoning_details"] and turn["content"] == "3"
    sure = {"role": "user", "content": "Are you sure? Think carefully."}
    follow = c.chat(model="m", max_tokens=200, reasoning=REASONING_ON, messages=STRAWBERRY + [turn, sure])
    assert "re-checked" in text_of(follow)
    without = c.chat(model="m", max_tokens=200, messages=STRAWBERRY + [{"role": "assistant", "content": "3"}, sure])
    assert "recount" in text_of(without)


def test_reasoning_can_eat_the_whole_budget(mock):
    llm = LLMClient(LLMConfig(model="m", reasoning=REASONING_ON), client=mock.client())
    with pytest.raises(TruncatedOutputError, match="reasoning used the whole budget"):
        llm.chat(STRAWBERRY, max_tokens=5)
    assert CALL_LOG[-1].reasoning_tokens == 5 and CALL_LOG[-1].status == "truncated"
    off = LLMClient(LLMConfig(model="m", reasoning=REASONING_OFF), client=mock.client())
    assert off.chat(STRAWBERRY, max_tokens=5) == "2"


def test_last_message_keeps_reasoning_details(mock):
    llm = LLMClient(LLMConfig(model="m", reasoning=REASONING_ON), client=mock.client())
    assert llm.chat(STRAWBERRY, max_tokens=200) == "3"
    assert llm.last_message["reasoning_details"][0]["text"].startswith("Spell it out")


def test_validation_repair_sends_reasoning_details_back():
    sent = []

    class Spy(FakeLLM):
        def chat(self, **body):
            sent.append(body["messages"])
            return super().chat(**body)

    bad = fake_response('{"answer": "x"}', reasoning_details=[{"type": "reasoning.text", "text": "thinking"}])
    good = fake_response(json.dumps({"answer": "3", "citations": ["results"], "confidence": "high"}))
    LLMClient(LLMConfig(model="m"), client=Spy([bad, good])).chat_structured(STRAWBERRY, Answer)
    assistant = sent[1][-2]
    assert assistant["role"] == "assistant" and assistant["reasoning_details"][0]["text"] == "thinking"


def test_streaming_over_http(mock):
    ask = [{"role": "user", "content": "In 3 sentences, what is sliding-window attention?"}]
    events = list(stream_chat(mock.client(), model="m", max_tokens=150, stream_options={"include_usage": True},
                              messages=ask))
    text = "".join((e.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for e in events)
    assert text.startswith("Sliding-window attention") and events[-1]["usage"]["completion_tokens"] > 0


def test_list_and_free_models(mock):
    ids = [m["id"] for m in list_models(mock.client())]
    assert "mock-llm" in ids                                         # not OpenRouter: no :free filter
    assert "mock-llm" in free_models(client=mock.client())


def test_rate_limit_message_reads_openrouter_headers(mock):
    with pytest.raises(RateLimitError) as e:
        mock.client().chat(model="m", max_tokens=5, messages=STRAWBERRY,
                           headers={"X-Mock-Status": "429", "X-Mock-Retry-After": "7"})
    assert e.value.response.headers["x-ratelimit-remaining"] == "0"
    assert "limit 20, remaining 0" in rate_limit_message(e.value) and "use_model" in rate_limit_message(e.value)


def test_bad_reasoning_type_is_400(mock):
    with pytest.raises(BadRequestError, match="reasoning"):
        mock.client().chat(model="m", max_tokens=5, messages=STRAWBERRY, reasoning="yes")


# --- Gemini: GEMINI_API_KEY ----------------------------------------------------------
def test_gemini_key_used_for_google_urls(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    h = OpenRouterClient().headers()
    assert h["x-goog-api-key"] == "AIza-test" and h["Authorization"] == "Bearer AIza-test"


def test_gemini_key_never_sent_to_openrouter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    h = OpenRouterClient().headers()
    assert h["Authorization"] == "Bearer sk-or-test" and "x-goog-api-key" not in h


def test_gemini_key_alone_defaults_to_gemini_url(monkeypatch):
    from paper_agent.config import load_env
    from paper_agent.llm_client import GEMINI_URL
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    assert OpenRouterClient().base_url == GEMINI_URL
    env = load_env()
    assert env["LLM_BASE_URL"] == GEMINI_URL and env["GEMINI_API_KEY"] == "AIza-test"


def test_missing_key_message_mentions_both(monkeypatch):
    from paper_agent.config import load_env
    for name in ("LLM_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=r"LLM_API_KEY \(or GEMINI_API_KEY\)"):
        load_env()


def test_reasoning_field_not_sent_to_google(monkeypatch):
    sent = {}

    def fake_post(self, path, body, headers=None, timeout=None):
        sent.update(body)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(OpenRouterClient, "post", fake_post)
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    LLMClient(LLMConfig(model="gemini-2.5-flash", reasoning=REASONING_OFF)).chat(STRAWBERRY)
    assert "reasoning" not in sent and sent["model"] == "gemini-2.5-flash"
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLMClient(LLMConfig(model="x:free", reasoning=REASONING_OFF)).chat(STRAWBERRY)
    assert sent["reasoning"] == {"enabled": False}
