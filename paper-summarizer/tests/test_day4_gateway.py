"""Day 4 in production mode: the LiteLLM proxy owns keys, budgets and rate limits; the api maps its errors to the
same friendly 429s as library mode. Needs the stack (`make test-gateway`), still offline: upstream is mock-llm."""
import os
import time

import httpx
import pytest

from paper_agent.fixtures.tinycoder_pdf import make_pdf
from paper_agent.service import keys

pytestmark = pytest.mark.gateway
API = os.environ.get("API_URL", "http://api:8000")


@pytest.fixture(scope="module")
def paper_id():
    keys.delete_keys(["test-gw-uploader"])
    key = keys.generate_key("test-gw-uploader", user_id="test-gw-uploader", max_budget=1.0)["key"]
    r = httpx.post(f"{API}/papers", headers={"Authorization": f"Bearer {key}"}, timeout=120,
                   files={"file": ("tinycoder.pdf", make_pdf(), "application/pdf")})
    assert r.status_code == 200, r.text
    yield r.json()["paper_id"]
    keys.delete_keys(["test-gw-uploader"])


@pytest.fixture
def make_key():
    made = []

    def make(alias, **settings):
        keys.delete_keys([alias])
        made.append(alias)
        return keys.generate_key(alias, user_id=alias, **settings)["key"]
    yield make
    keys.delete_keys(made)


def ask(key, paper_id, question="How much faster is TinyCoder than the 7B baseline?"):
    return httpx.post(f"{API}/papers/{paper_id}/ask", headers={"Authorization": f"Bearer {key}"},
                      json={"question": question}, timeout=120)


def test_key_generation(make_key):
    key = make_key("test-gw-carol", max_budget=0.5, budget_duration="1d", rpm_limit=7, tpm_limit=50000)
    info = keys.key_info(key)
    assert key.startswith("sk-") and info["user_id"] == "test-gw-carol" and info["key_alias"] == "test-gw-carol"
    assert (info["max_budget"], info["budget_duration"], info["rpm_limit"]) == (0.5, "1d", 7)
    assert set(keys.MODELS) <= set(info["models"])


def test_unknown_key_is_401(paper_id):
    r = ask("sk-not-a-real-key", paper_id)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"


def test_budget_exhaustion_is_the_same_friendly_429(make_key, paper_id):
    key = make_key("test-gw-bob", max_budget=0.0005, budget_duration="1d", rpm_limit=100)
    codes = []
    for _ in range(5):
        r = ask(key, paper_id)
        codes.append(r.status_code)
        if r.status_code != 200:
            break
    assert codes[0] == 200 and codes[-1] == 429, codes
    body = r.json()
    assert body["error"] == "budget_exceeded" and "resets at midnight" in body["message"]
    assert "Traceback" not in r.text and "sk-" not in r.text


def test_rate_limit_comes_back_with_retry_after(make_key, paper_id):
    key = make_key("test-gw-dave", max_budget=1.0, rpm_limit=3)
    codes = []
    for _ in range(4):
        r = ask(key, paper_id)
        codes.append(r.status_code)
        if r.status_code == 429:
            break
    assert codes[-1] == 429, codes
    assert r.json()["error"] == "rate_limited" and int(r.headers["Retry-After"]) > 0


def test_me_usage_matches_key_info(make_key, paper_id):
    key = make_key("test-gw-erin", max_budget=1.0, rpm_limit=100)
    assert ask(key, paper_id).status_code == 200
    deadline = time.monotonic() + 90                   # the proxy writes spend to Postgres in batches
    while keys.key_info(key)["spend"] == 0 and time.monotonic() < deadline:
        time.sleep(3)
    usage = httpx.get(f"{API}/me/usage", headers={"Authorization": f"Bearer {key}"}, timeout=30).json()
    info = keys.key_info(key)
    assert usage["user"] == "test-gw-erin" and usage["source"] == "gateway"
    assert usage["spent_usd"] == round(info["spend"], 6) > 0 and usage["budget_usd"] == info["max_budget"] == 1.0
