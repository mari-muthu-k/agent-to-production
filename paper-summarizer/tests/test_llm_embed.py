"""Day 2 additions to llm_client: embed() shares chat()'s protections; Gemini reasoning; provider-neutral
error handling (list-shaped bodies, daily caps); truthful logs when a provider reports no usage."""
import json
import threading
import time

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from paper_agent.llm_client import (
    CALL_LOG,
    DAILY_CAP_MESSAGE,
    BudgetExceededError,
    CircuitOpenError,
    DailyQuotaError,
    LLMClient,
    LLMConfig,
    OpenRouterClient,
    RateLimitError,
    ServerError,
    TransientError,
    api_error,
    summarize_calls,
)
from paper_agent.testing import FakeLLM, fake_embedding_response, fake_response

FAST = dict(model="fake", embed_model="fake-embed", base_delay_s=0.01, max_delay_s=0.1)
GEMINI_DAILY_429 = [{"error": {
    "code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "You exceeded your current quota, please check your plan and billing details.",
    "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                         "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31387s"}]}}]


def http_response(status: int, body, headers=None) -> requests.Response:
    r = requests.Response()
    r.status_code, r.headers = status, CaseInsensitiveDict(headers or {})
    r._content = json.dumps(body).encode()
    return r


# --- embed() ------------------------------------------------------------------------
def test_embed_returns_vectors_in_order_from_one_request():
    fake = FakeLLM([{"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [3.0, 4.0]}],
                     "usage": {"prompt_tokens": 7}}])
    vecs = LLMClient(LLMConfig(**FAST), client=fake).embed(["first", "second"])
    assert vecs == [[3.0, 4.0], [0.0, 2.0]]                  # sorted by index, NOT normalised
    assert fake.calls == 1 and fake.bodies[0]["input"] == ["first", "second"]
    rec = CALL_LOG[-1]
    assert (rec.kind, rec.model, rec.input_tokens, rec.output_tokens, rec.usage_reported) == \
        ("embed", "fake-embed", 7, 0, True)


def test_embed_model_comes_from_env_at_every_call(monkeypatch):
    fake = FakeLLM([fake_embedding_response([[1.0]])] * 2)
    llm = LLMClient(LLMConfig(), client=fake)
    monkeypatch.setenv("EMBED_MODEL", "embed-a")
    llm.embed(["x"])
    monkeypatch.setenv("EMBED_MODEL", "embed-b")
    llm.embed(["x"])
    assert [b["model"] for b in fake.bodies] == ["embed-a", "embed-b"]


def test_embed_without_a_model_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    with pytest.raises(ValueError, match="EMBED_MODEL"):
        LLMClient(LLMConfig(), client=FakeLLM([])).embed(["x"])


def test_embed_batches_large_inputs():
    fake = FakeLLM([fake_embedding_response([[1.0]] * 2), fake_embedding_response([[1.0]] * 2),
                    fake_embedding_response([[1.0]])])
    vecs = LLMClient(LLMConfig(**FAST, embed_batch_size=2), client=fake).embed(list("abcde"))
    assert len(vecs) == 5 and fake.calls == 3 and len(CALL_LOG) == 3


def test_embed_sends_timeout_s():
    fake = FakeLLM([fake_embedding_response([[1.0]])])
    LLMClient(LLMConfig(**FAST, timeout_s=12.5), client=fake).embed(["x"])
    assert fake.bodies[0]["timeout"] == 12.5


def test_embed_without_usage_logs_null_not_zero():
    LLMClient(LLMConfig(**FAST), client=FakeLLM([fake_embedding_response([[1.0]], usage=False)])).embed(["x"])
    rec = CALL_LOG[-1]
    assert rec.input_tokens is None and rec.usage_reported is False and rec.output_tokens == 0
    assert summarize_calls()["usage_unreported"] == 1 and summarize_calls()["input_tokens"] == 0


def test_embed_retries_transient_errors_with_backoff(no_sleep):
    fake = FakeLLM([TransientError(), fake_embedding_response([[1.0]])])
    LLMClient(LLMConfig(**FAST), client=fake).embed(["x"])
    assert fake.calls == 2 and len(no_sleep) == 1 and CALL_LOG[-1].attempts == 2


def test_embed_failed_call_is_logged(no_sleep):
    fake = FakeLLM([TransientError()] * 5)
    with pytest.raises(TransientError):
        LLMClient(LLMConfig(**FAST, max_retries=2), client=fake).embed(["x"])
    assert CALL_LOG[-1].status == "failed" and CALL_LOG[-1].kind == "embed" and CALL_LOG[-1].attempts == 3


def test_embed_respects_the_budget_guard():
    fake = FakeLLM([fake_response("ok"), fake_embedding_response([[1.0]])])
    llm = LLMClient(LLMConfig(**FAST, price_in_per_1m=1000, price_out_per_1m=1000, budget_usd=0.01), client=fake)
    llm.chat([{"role": "user", "content": "hi"}])             # $0.12 spent: over budget
    with pytest.raises(BudgetExceededError):
        llm.embed(["x"])
    assert fake.calls == 1


def test_embed_shares_the_circuit_breaker_with_chat(no_sleep):
    fake = FakeLLM([TransientError()] * 2)
    llm = LLMClient(LLMConfig(**FAST, max_retries=1, breaker_threshold=2), client=fake)
    with pytest.raises(TransientError):
        llm.embed(["x"])                                       # 2 failures: circuit opens
    with pytest.raises(CircuitOpenError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert fake.calls == 2


def test_embed_concurrency_cap_holds():
    lock, state = threading.Lock(), {"now": 0, "max": 0}

    class SlowProvider:
        def embeddings(self, **body):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return fake_embedding_response([[1.0]])

    llm = LLMClient(LLMConfig(**FAST, max_concurrency=2), client=SlowProvider())
    threads = [threading.Thread(target=llm.embed, args=(["x"],)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] <= 2


def test_embed_against_mock_llm(mock):
    llm = LLMClient(LLMConfig(embed_model="mock-embed"), client=mock.client())
    vecs = llm.embed(["sliding-window attention", "pass@1 on long files"])
    assert len(vecs) == 2 and len(vecs[0]) == len(vecs[1]) > 0
    assert CALL_LOG[-1].kind == "embed" and CALL_LOG[-1].input_tokens > 0


# --- Gemini reasoning ------------------------------------------------------------------
def sent_body(monkeypatch, base_url: str, model: str, **body) -> dict:
    client = OpenRouterClient(api_key="k", base_url=base_url)
    seen = {}
    monkeypatch.setattr(client, "post", lambda path, b, headers=None, timeout=None: seen.update(b) or {})
    client.chat(model=model, messages=[], **body)
    return seen


GOOGLE = "https://generativelanguage.googleapis.com/v1beta/openai"


def test_gemini_model_gets_reasoning_effort_none(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    body = sent_body(monkeypatch, GOOGLE, "gemini-3.8-flash", reasoning={"enabled": False})
    assert body["reasoning_effort"] == "none" and "reasoning" not in body


def test_reasoning_effort_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    assert sent_body(monkeypatch, GOOGLE, "gemini-3.8-flash")["reasoning_effort"] == "low"
    monkeypatch.setenv("LLM_REASONING_EFFORT", "default")
    assert "reasoning_effort" not in sent_body(monkeypatch, GOOGLE, "gemini-3.8-flash")


def test_non_gemini_model_gets_no_reasoning_parameter(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    body = sent_body(monkeypatch, "http://mock-llm:8000/v1", "mock-llm")
    assert "reasoning_effort" not in body and "reasoning" not in body


def test_gemini_on_openrouter_uses_openrouters_reasoning_object(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    body = sent_body(monkeypatch, "https://openrouter.ai/api/v1", "google/gemini-3.8-flash")
    assert body["reasoning"] == {"enabled": False} and "reasoning_effort" not in body


def test_reasoning_effort_matches_the_openai_sdk():
    """The parameter name and value the official SDK sends to OpenAI-compatible endpoints."""
    import inspect
    import typing

    from openai.resources.chat.completions import Completions
    from openai.types.shared.reasoning_effort import ReasoningEffort
    assert "reasoning_effort" in inspect.signature(Completions.create).parameters
    literal = [a for a in typing.get_args(ReasoningEffort) if typing.get_args(a)][0]
    assert "none" in typing.get_args(literal)


# --- provider-neutral errors -------------------------------------------------------------
def test_list_shaped_error_body_parses():
    e = api_error(http_response(503, [{"error": {"code": 503, "message": "This model is currently experiencing "
                                                 "high demand.", "status": "UNAVAILABLE"}}]),
                  [{"error": {"code": 503, "message": "This model is currently experiencing high demand.",
                              "status": "UNAVAILABLE"}}])
    assert isinstance(e, ServerError) and "high demand" in e.message and "[{" not in e.message


def test_list_shaped_daily_429_is_a_daily_quota_error():
    e = api_error(http_response(429, GEMINI_DAILY_429), GEMINI_DAILY_429)
    assert isinstance(e, DailyQuotaError) and isinstance(e, RateLimitError)
    assert DAILY_CAP_MESSAGE in e.message


def test_per_minute_429_is_not_a_daily_cap():
    body = [{"error": {"code": 429, "message": "slow down", "details": [{"retryDelay": "20s"}]}}]
    e = api_error(http_response(429, body), body)
    assert type(e) is RateLimitError


@pytest.mark.parametrize("call", ["chat", "embed"])
def test_daily_cap_is_not_retried_for_hours(no_sleep, call):
    capped = api_error(http_response(429, GEMINI_DAILY_429), GEMINI_DAILY_429)
    fake = FakeLLM([capped] + [fake_response("never")] * 3)
    llm = LLMClient(LLMConfig(**FAST, max_retries=3), client=fake)
    t = time.perf_counter()
    with pytest.raises(DailyQuotaError, match="daily free-tier limit reached"):
        llm.chat([{"role": "user", "content": "hi"}]) if call == "chat" else llm.embed(["x"])
    assert fake.calls == 1 and no_sleep == [] and time.perf_counter() - t < 1


def test_header_detected_daily_cap_reads_plainly(no_sleep):
    reset_in_hours = str(int((time.time() + 6 * 3600) * 1000))
    capped = RateLimitError("HTTP 429", status_code=429, response=http_response(
        429, {}, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset_in_hours}))
    with pytest.raises(DailyQuotaError, match="switch to LLM_FALLBACK_MODEL or another key"):
        LLMClient(LLMConfig(**FAST), client=FakeLLM([capped])).chat([{"role": "user", "content": "hi"}])
