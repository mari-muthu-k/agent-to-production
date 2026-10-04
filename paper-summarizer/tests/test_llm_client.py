"""The Day 1 reliability rules, tested offline with fakes and against mock-llm."""
import threading
import time

import httpx2
import openai
import pytest

from paper_agent.llm_client import (
    CALL_LOG,
    RETRYABLE_ERRORS,
    BudgetExceededError,
    CircuitOpenError,
    LLMClient,
    LLMConfig,
    StructuredOutputError,
    TransientError,
    TruncatedOutputError,
    summarize_calls,
)
from paper_agent.schemas import Answer
from paper_agent.testing import FakeLLM, FlakyProvider, fake_response

MSG = [{"role": "user", "content": "In one sentence, what is pass@1?"}]
FAST = dict(model="fake", base_delay_s=0.01, max_delay_s=0.1)


def status_error(cls, status: int, headers=None):
    request = httpx2.Request("POST", "http://test/v1/chat/completions")
    response = httpx2.Response(status, headers=headers or {}, request=request)
    return cls(f"HTTP {status}", response=response, body=None)


# --- backoff -----------------------------------------------------------------
def test_backoff_grows_exponentially():
    c = LLMClient(LLMConfig(**FAST), client=FakeLLM([]))
    maxes = [max(c._backoff_delay(a) for _ in range(300)) for a in range(4)]
    assert maxes[0] < maxes[1] < maxes[2] < maxes[3]


def test_backoff_is_jittered_and_capped():
    c = LLMClient(LLMConfig(model="fake", base_delay_s=1.0, max_delay_s=5.0), client=FakeLLM([]))
    samples = [c._backoff_delay(2) for _ in range(50)]
    assert len({round(s, 6) for s in samples}) > 1, "no jitter"
    assert all(0 <= s <= 4.0 for s in samples)
    assert all(0 <= c._backoff_delay(10) <= 5.0 for _ in range(50)), "not capped at max_delay_s"


# --- retries -----------------------------------------------------------------
def test_transient_failures_are_retried(no_sleep):
    fake = FakeLLM([TransientError(), TransientError(), fake_response("ok")])
    assert LLMClient(LLMConfig(**FAST), client=fake).chat(MSG) == "ok"
    assert fake.calls == 3 and len(no_sleep) == 2
    assert CALL_LOG[-1].attempts == 3 and CALL_LOG[-1].status == "ok"


def test_retries_stop_after_max_retries(no_sleep):
    fake = FakeLLM([TransientError()] * 10)
    with pytest.raises(TransientError):
        LLMClient(LLMConfig(**FAST, max_retries=2, breaker_threshold=99), client=fake).chat(MSG)
    assert fake.calls == 3   # 1 try + 2 retries
    assert CALL_LOG[-1].status == "failed"


def test_retry_after_from_exception_attribute(no_sleep):
    fake = FakeLLM([TransientError(retry_after=3), fake_response("ok")])
    LLMClient(LLMConfig(model="fake", base_delay_s=0.01, max_delay_s=20), client=fake).chat(MSG)
    assert no_sleep == [3.0]


def test_retry_after_from_http_header(no_sleep):
    err = status_error(openai.RateLimitError, 429, {"retry-after": "4"})
    fake = FakeLLM([err, fake_response("ok")])
    LLMClient(LLMConfig(model="fake", base_delay_s=0.01, max_delay_s=20), client=fake).chat(MSG)
    assert no_sleep == [4.0]


def test_retry_after_is_capped_by_max_delay(no_sleep):
    fake = FakeLLM([TransientError(retry_after=600), fake_response("ok")])
    LLMClient(LLMConfig(model="fake", base_delay_s=0.01, max_delay_s=20), client=fake).chat(MSG)
    assert no_sleep == [20.0]


def test_retry_after_header_from_mock_llm(mock, no_sleep):
    """A real 429 with a Retry-After header, end to end through the OpenAI SDK."""
    mock.fault(status=429, retry_after=2, count=1)
    llm = LLMClient(LLMConfig(model="mock-llm", base_delay_s=0.01, max_delay_s=20), client=mock.client())
    assert llm.chat(MSG, max_tokens=60)
    assert no_sleep == [2.0]
    assert mock.stats()["status_429"] == 1


@pytest.mark.parametrize("cls,status", [(openai.BadRequestError, 400), (openai.AuthenticationError, 401),
                                        (openai.NotFoundError, 404)])
def test_client_errors_are_not_retried(cls, status, no_sleep):
    fake = FakeLLM([status_error(cls, status), fake_response("never reached")])
    with pytest.raises(cls):
        LLMClient(LLMConfig(**FAST), client=fake).chat(MSG)
    assert fake.calls == 1 and no_sleep == []


def test_400_from_mock_llm_is_not_retried(mock, no_sleep):
    llm = LLMClient(LLMConfig(model="mock-llm", **{k: v for k, v in FAST.items() if k != "model"}),
                    client=mock.client())
    with pytest.raises(openai.BadRequestError):
        llm.chat(MSG, temperature=5.0)
    assert mock.stats()["chat_requests"] == 1 and no_sleep == []


def test_retryable_errors_are_exactly_timeouts_connection_429_5xx():
    assert set(RETRYABLE_ERRORS) == {openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
                                     openai.InternalServerError, TransientError}


# --- circuit breaker ---------------------------------------------------------
def test_breaker_opens_and_fails_fast():
    fake = FakeLLM([TransientError()] * 10)
    llm = LLMClient(LLMConfig(**FAST, max_retries=0, breaker_threshold=3, breaker_reset_s=60), client=fake)
    for _ in range(3):
        with pytest.raises(TransientError):
            llm.chat(MSG)
    assert llm.breaker.state == "open"
    t = time.perf_counter()
    with pytest.raises(CircuitOpenError):
        llm.chat(MSG)
    assert time.perf_counter() - t < 0.05
    assert fake.calls == 3, "an open circuit must not reach the provider"


def test_breaker_half_open_trial_closes_on_success():
    fake = FakeLLM([TransientError(), TransientError(), fake_response("recovered")])
    llm = LLMClient(LLMConfig(**FAST, max_retries=0, breaker_threshold=2, breaker_reset_s=0.05), client=fake)
    for _ in range(2):
        with pytest.raises(TransientError):
            llm.chat(MSG)
    assert llm.breaker.state == "open"
    time.sleep(0.06)
    assert llm.breaker.state == "half_open"
    assert llm.chat(MSG) == "recovered" and llm.breaker.state == "closed"


# --- fallback ----------------------------------------------------------------
def test_fallback_model_is_used(no_sleep):
    primary = FakeLLM([TransientError()] * 5)
    backup = FakeLLM([fake_response("from fallback")])
    llm = LLMClient(LLMConfig(**{**FAST, "fallback_model": "backup-model"}, max_retries=1),
                    client=primary, fallback_client=backup)
    assert llm.chat(MSG) == "from fallback"
    assert primary.calls == 2 and backup.calls == 1
    assert CALL_LOG[-1].status == "fallback_ok" and CALL_LOG[-1].model == "backup-model"


def test_fallback_not_used_for_client_errors():
    primary = FakeLLM([status_error(openai.BadRequestError, 400)])
    backup = FakeLLM([fake_response("should not be called")])
    llm = LLMClient(LLMConfig(**{**FAST, "fallback_model": "backup"}), client=primary, fallback_client=backup)
    with pytest.raises(openai.BadRequestError):
        llm.chat(MSG)
    assert backup.calls == 0


def test_fallback_against_mock_llm(mock, no_sleep):
    real = mock.client()
    llm = LLMClient(LLMConfig(model="mock-llm", fallback_model="mock-llm-fallback", max_retries=1),
                    client=FlakyProvider(real, failures=100), fallback_client=real)
    assert "attention" in llm.chat([{"role": "user", "content": "In one sentence, what is sliding-window "
                                                                "attention?"}], max_tokens=80).lower()
    assert CALL_LOG[-1].model == "mock-llm-fallback"


# --- budget ------------------------------------------------------------------
def test_budget_guard_stops_calls():
    fake = FakeLLM([fake_response("ok")] * 10)   # each: 100 in + 20 out tokens
    llm = LLMClient(LLMConfig(model="fake", price_in_per_1m=1000, price_out_per_1m=1000, budget_usd=0.25),
                    client=fake)
    llm.chat(MSG)   # $0.12
    llm.chat(MSG)   # $0.24
    llm.chat(MSG)   # $0.36: allowed, because the guard checks before the call
    with pytest.raises(BudgetExceededError):
        llm.chat(MSG)
    assert fake.calls == 3


# --- concurrency -------------------------------------------------------------
def test_concurrency_cap_holds():
    lock, state = threading.Lock(), {"now": 0, "max": 0}

    class SlowProvider:
        def __init__(self):
            from types import SimpleNamespace as NS
            self.chat = NS(completions=NS(create=self._create))

        def _create(self, **kwargs):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return fake_response("ok")

    llm = LLMClient(LLMConfig(model="fake", max_concurrency=4), client=SlowProvider())
    threads = [threading.Thread(target=llm.chat, args=(MSG,)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 4
    assert len(CALL_LOG) == 20


# --- truncation --------------------------------------------------------------
def test_truncation_raises():
    fake = FakeLLM([fake_response('{"headline": "Tiny', finish_reason="length")])
    with pytest.raises(TruncatedOutputError):
        LLMClient(LLMConfig(**FAST), client=fake).chat(MSG)
    assert CALL_LOG[-1].status == "truncated"


def test_truncation_from_mock_llm(mock):
    llm = LLMClient(LLMConfig(model="mock-llm"), client=mock.client())
    with pytest.raises(TruncatedOutputError):
        llm.chat([{"role": "user", "content": "Explain sliding-window attention in one paragraph."}], max_tokens=12)


def test_max_tokens_param_is_configurable():
    seen = {}

    class Spy(FakeLLM):
        def _create(self, **kwargs):
            seen.update(kwargs)
            return super()._create(**kwargs)

    LLMClient(LLMConfig(model="fake", max_tokens_param="max_completion_tokens"),
              client=Spy([fake_response("ok")])).chat(MSG, max_tokens=42, temperature=None)
    assert seen["max_completion_tokens"] == 42 and "max_tokens" not in seen and "temperature" not in seen


# --- structured output -------------------------------------------------------
GOOD = '{"answer": "41% vs 42%", "citations": ["results"], "confidence": "high"}'


def test_validation_repair_works():
    fake = FakeLLM([fake_response('Sure! {"answer": "x", "citations": [], "confidence": "certain"}'),
                    fake_response("```json\n" + GOOD + "\n```")])
    result = LLMClient(LLMConfig(**FAST), client=fake).chat_structured(MSG, Answer)
    assert result.citations == ["results"] and fake.calls == 2


def test_validation_fails_loudly_after_one_repair():
    fake = FakeLLM([fake_response('{"answer": 1}'), fake_response('{"answer": 2}'), fake_response(GOOD)])
    with pytest.raises(StructuredOutputError):
        LLMClient(LLMConfig(**FAST), client=fake).chat_structured(MSG, Answer)
    assert fake.calls == 2


def test_validation_repair_against_mock_llm(mock):
    client = mock.client(default_headers={"X-Mock-Bad-Json": "1"})
    llm = LLMClient(LLMConfig(model="mock-llm"), client=client)
    from paper_agent.explain import paper_messages
    from paper_agent.schemas import PaperExplainer
    result = llm.chat_structured(paper_messages(), PaperExplainer, max_tokens=900)
    assert result.caveats and len(CALL_LOG) == 2


# --- logging -----------------------------------------------------------------
def test_one_record_per_call_and_summary(no_sleep):
    llm = LLMClient(LLMConfig(**FAST), client=FakeLLM([TransientError(), fake_response("a"), fake_response("b")]))
    llm.chat(MSG)
    llm.chat(MSG)
    s = summarize_calls()
    assert s["calls"] == 2 and s["retries"] == 1 and s["input_tokens"] == 200
