"""Day 4, Section 5: the gateway (fallback, one retry layer), budgets and rate limits. Offline: mock-llm."""
import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from paper_agent.service import app as service
from paper_agent.service import usage
from paper_agent.service.errors import BudgetExceededError
from paper_agent.service.gateway import make_chat_model, make_router

HELLO = [{"role": "user", "content": "Reply with one word: ready"}]


def test_fallback_answers_with_the_fallback_deployment_in_two_calls(day4, mock):
    router = make_router(primary="this-model-does-not-exist")
    before = day4.calls()
    reply = make_chat_model(router).invoke(HELLO)
    assert reply.response_metadata["model_id"] == "fallback"
    assert day4.calls() - before == 2                    # the 404, then the fallback: no retry of a 404


def test_a_5xx_is_retried_twice_by_litellm_only(day4, mock):
    mock.fault(status=500, count=10, model="mock-llm")
    router = make_router(fallback=False)                 # num_retries=2
    before = day4.calls()
    with pytest.raises(openai.InternalServerError):        # LiteLLM's subclasses it
        router.completion(model="paper-explainer", messages=HELLO, max_tokens=5)
    assert day4.calls() - before == 3                    # 1 attempt + 2 retries


def test_langchain_adds_no_retries_on_top(day4, mock):
    mock.fault(status=500, count=10, model="mock-llm")
    model = make_chat_model(make_router(fallback=False))  # ChatLiteLLMRouter(max_retries=1): one attempt
    before = day4.calls()
    with pytest.raises(openai.InternalServerError):        # LiteLLM's subclasses it
        model.invoke(HELLO)
    assert day4.calls() - before == 3                    # still 3, not 3 x 3


def test_check_budget_raises_at_or_over_the_budget_not_under(day4):
    usage.BUDGETS["bob"] = 0.001
    usage.SPEND["bob"] = 0.000999
    usage.check_budget("bob")                            # under: fine
    for spent in (0.001, 0.002):
        usage.SPEND["bob"] = spent
        with pytest.raises(BudgetExceededError):
            usage.check_budget("bob")
    usage.check_budget("someone-new")                    # default budget, nothing spent


def test_budget_guard_stops_a_run_before_the_next_model_call(day4):
    from paper_agent.agent.fakes import FakeToolModel
    from paper_agent.service.pipeline import Ctx, build_agent
    script = [AIMessage("", tool_calls=[{"name": "search_paper", "args": {"query": "speed"}, "id": "t1"}]),
              AIMessage("", tool_calls=[{"name": "search_paper", "args": {"query": "limits"}, "id": "t2"}]),
              AIMessage("Done [p3-c1].")]
    fake = FakeToolModel(script=script)
    agent = build_agent(model=fake, extra_middleware=[usage.budget_guard])
    usage.BUDGETS["bob"] = 1e-9                          # the first call is allowed, and spends more than this
    with pytest.raises(BudgetExceededError):
        agent.invoke({"messages": [{"role": "user", "content": "How fast is it?"}]},
                     context=Ctx("bob", day4.paper_id))
    assert len(fake.requests) == 1 and usage.CALLS["bob"] == 1 and usage.SPEND["bob"] > usage.BUDGETS["bob"]


@pytest.fixture
def client(day4):
    from paper_agent.service.pipeline import build_agent
    app = service.build_app(build_agent(router=day4.router, extra_middleware=[usage.budget_guard]), day4.store)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_over_budget_is_a_friendly_429(client, day4):
    usage.BUDGETS["bob"] = 0.0
    r = client.post(f"/papers/{day4.paper_id}/ask", headers={"Authorization": "Bearer key-bob"},
                    json={"question": "What is TinyCoder?"})
    assert r.status_code == 429
    assert r.json()["error"] == "budget_exceeded" and "resets at midnight" in r.json()["message"]
    alice = client.post(f"/papers/{day4.paper_id}/ask", headers={"Authorization": "Bearer key-alice"},
                        json={"question": "What is TinyCoder?"})
    assert alice.status_code == 200                      # budgets are per user


def test_the_request_over_the_rate_limit_gets_429_and_retry_after(client):
    service.set_rate_limit("alice", 3)
    codes = [client.get("/me/usage", headers={"Authorization": "Bearer key-alice"}) for _ in range(5)]
    assert [r.status_code for r in codes] == [200, 200, 200, 429, 429]
    assert codes[3].json()["error"] == "rate_limited" and 55 <= int(codes[3].headers["Retry-After"]) <= 60
    assert client.get("/me/usage", headers={"Authorization": "Bearer key-bob"}).status_code == 200   # per user
    service.set_rate_limit("alice", None)
    assert client.get("/me/usage", headers={"Authorization": "Bearer key-alice"}).status_code == 200


def test_me_usage_reports_spend_and_budget(client, day4):
    usage.BUDGETS["bob"] = 0.0002
    usage.SPEND["bob"], usage.CALLS["bob"] = 0.00025, 4
    r = client.get("/me/usage", headers={"Authorization": "Bearer key-bob"}).json()
    assert r == {"user": "bob", "llm_calls": 4, "spent_usd": 0.00025, "budget_usd": 0.0002, "remaining_usd": -5e-05}
