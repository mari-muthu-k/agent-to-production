"""The Day 1 reliability rules, tested offline with fakes and against mock-llm."""
import threading
import time

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from paper_agent.llm_client import (
    CALL_LOG,
    RETRYABLE_ERRORS,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    CircuitOpenError,
    LLMClient,
    LLMConfig,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    ServerError,
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
    response = requests.Response()
    response.status_code, response.headers = status, CaseInsensitiveDict(headers or {})
    return cls(f"HTTP {status}", status_code=status, response=response)


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
    err = status_error(RateLimitError, 429, {"retry-after": "4"})
    fake = FakeLLM([err, fake_response("ok")])
    LLMClient(LLMConfig(model="fake", base_delay_s=0.01, max_delay_s=20), client=fake).chat(MSG)
    assert no_sleep == [4.0]


def test_retry_after_is_capped_by_max_delay(no_sleep):
    fake = FakeLLM([TransientError(retry_after=600), fake_response("ok")])
    LLMClient(LLMConfig(model="fake", base_delay_s=0.01, max_delay_s=20), client=fake).chat(MSG)
    assert no_sleep == [20.0]


def test_retry_after_header_from_mock_llm(mock, no_sleep):
    """A real 429 with a Retry-After header, end to end over HTTP."""
    mock.fault(status=429, retry_after=2, count=1)
    llm = LLMClient(LLMConfig(model="mock-llm", base_delay_s=0.01, max_delay_s=20), client=mock.client())
    assert llm.chat(MSG, max_tokens=60)
    assert no_sleep == [2.0]
    assert mock.stats()["status_429"] == 1


@pytest.mark.parametrize("cls,status", [(BadRequestError, 400), (AuthenticationError, 401),
                                        (PaymentRequiredError, 402), (NotFoundError, 404)])
def test_client_errors_are_not_retried(cls, status, no_sleep):
    fake = FakeLLM([status_error(cls, status), fake_response("never reached")])
    with pytest.raises(cls):
        LLMClient(LLMConfig(**FAST), client=fake).chat(MSG)
    assert fake.calls == 1 and no_sleep == []


def test_400_from_mock_llm_is_not_retried(mock, no_sleep):
    llm = LLMClient(LLMConfig(model="mock-llm", **{k: v for k, v in FAST.items() if k != "model"}),
                    client=mock.client())
    with pytest.raises(BadRequestError):
        llm.chat(MSG, temperature=5.0)
    assert mock.stats()["chat_requests"] == 1 and no_sleep == []


def test_retryable_errors_are_exactly_timeouts_connection_429_5xx():
    assert set(RETRYABLE_ERRORS) == {APITimeoutError, APIConnectionError, RateLimitError, ServerError,
                                     TransientError}


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
    primary = FakeLLM([status_error(BadRequestError, 400)])
    backup = FakeLLM([fake_response("should not be called")])
    llm = LLMClient(LLMConfig(**{**FAST, "fallback_model": "backup"}), client=primary, fallback_client=backup)
    with pytest.raises(BadRequestError):
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
        def chat(self, **kwargs):
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
        def chat(self, **kwargs):
            seen.update(kwargs)
            return super().chat(**kwargs)

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
    client = mock.client(headers={"X-Mock-Bad-Json": "1"})
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


# --- switching models (free tiers have caps) -------------------------------------
def test_models_from_env_are_read_at_every_call(monkeypatch):
    seen = []

    class Spy(FakeLLM):
        def chat(self, **kwargs):
            seen.append(kwargs["model"])
            return super().chat(**kwargs)

    llm = LLMClient(LLMConfig(), client=Spy([fake_response("a"), fake_response("b")]))
    monkeypatch.setenv("LLM_MODEL", "first-model:free")
    llm.chat(MSG)
    monkeypatch.setenv("LLM_MODEL", "second-model:free")          # switched mid-session, no restart
    llm.chat(MSG)
    assert seen == ["first-model:free", "second-model:free"]


def test_comma_list_falls_through_on_retryable_errors(monkeypatch, no_sleep):
    monkeypatch.setenv("LLM_MODEL", "busy:free,spare:free")
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    fake = FakeLLM([status_error(ServerError, 503)] * 2 + [fake_response("from spare")])
    llm = LLMClient(LLMConfig(max_retries=1, base_delay_s=0.01), client=fake)
    assert llm.chat(MSG) == "from spare"
    assert CALL_LOG[-1].model == "spare:free" and CALL_LOG[-1].status == "fallback_ok"
    assert llm.exhausted == set()                     # a busy model is not "exhausted": tried again next time


def test_daily_cap_switches_immediately_and_is_remembered(monkeypatch, no_sleep):
    monkeypatch.setenv("LLM_MODEL", "capped:free,spare:free")
    reset_in_hours = str(int((time.time() + 6 * 3600) * 1000))
    capped = status_error(RateLimitError, 429, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset_in_hours})
    seen = []

    class Spy(FakeLLM):
        def chat(self, **kwargs):
            seen.append(kwargs["model"])
            return super().chat(**kwargs)

    llm = LLMClient(LLMConfig(max_retries=3), client=Spy([capped, fake_response("a"), fake_response("b")]))
    assert llm.chat(MSG) == "a" and no_sleep == []           # no pointless waiting on a daily cap
    assert llm.exhausted == {"capped:free"}
    llm.chat(MSG)
    assert seen == ["capped:free", "spare:free", "spare:free"]


def test_out_of_credits_402_switches_model(monkeypatch, no_sleep):
    monkeypatch.setenv("LLM_MODEL", "paid-model,free-model:free")
    fake = FakeLLM([status_error(PaymentRequiredError, 402), fake_response("free answer")])
    llm = LLMClient(LLMConfig(), client=fake)
    assert llm.chat(MSG) == "free answer" and llm.exhausted == {"paid-model"}


def test_no_model_configured_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    with pytest.raises(ValueError, match="set LLM_MODEL"):
        LLMClient(LLMConfig(), client=FakeLLM([])).chat(MSG)


def test_http_errors_map_to_classes(mock):
    client = mock.client()
    for status, cls in [(400, BadRequestError), (401, AuthenticationError), (404, NotFoundError),
                        (429, RateLimitError), (503, ServerError)]:
        with pytest.raises(cls) as e:
            client.chat(model="m", max_tokens=5, messages=MSG, headers={"X-Mock-Status": str(status)})
        assert e.value.status_code == status
    with pytest.raises(APITimeoutError):
        mock.client(timeout=0.2).chat(model="m", max_tokens=5, messages=MSG, headers={"X-Mock-Delay-Ms": "2000"})
    with pytest.raises(APIConnectionError):
        type(client)(api_key="k", base_url="http://127.0.0.1:9/v1").chat(model="m", messages=MSG)
