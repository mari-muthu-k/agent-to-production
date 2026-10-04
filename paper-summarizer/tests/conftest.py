"""Shared fixtures. No API keys: unit tests use fakes; integration tests use mock-llm.

In Docker (`make test`) MOCK_BASE_URL points at the mock-llm service. Run outside Docker,
the mock is started in-process on a free port instead.
"""
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from paper_agent.llm_client import CALL_LOG

ROOT = Path(__file__).resolve().parents[1]
MOCK_KEY = "sk-mock-local"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_url():
    """Base URL of a running mock-llm (without /v1)."""
    url = os.environ.get("MOCK_BASE_URL")
    if url:
        httpx.get(f"{url}/health", timeout=5).raise_for_status()
        yield url.rstrip("/")
        return
    import uvicorn

    from mock_llm.server import app
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture
def mock(mock_url):
    """Clean mock state (faults, stats) for each test; returns a small control helper."""
    ctl = MockControl(mock_url)
    ctl.reset()
    yield ctl
    ctl.reset()


class MockControl:
    def __init__(self, url):
        self.url = url
        self.v1 = f"{url}/v1"

    def reset(self):
        httpx.post(f"{self.url}/mock/reset", timeout=5).raise_for_status()

    def fault(self, **fault):
        httpx.post(f"{self.url}/mock/faults", json=fault, timeout=5).raise_for_status()

    def stats(self) -> dict:
        return httpx.get(f"{self.url}/mock/stats", timeout=5).json()

    def client(self, **kwargs) -> OpenAI:
        kwargs.setdefault("api_key", MOCK_KEY)
        kwargs.setdefault("max_retries", 0)   # tests check our retries, never the SDK's
        return OpenAI(base_url=self.v1, **kwargs)


@pytest.fixture(autouse=True)
def _clear_call_log():
    CALL_LOG.clear()
    yield
    CALL_LOG.clear()


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff sleeps instead of waiting."""
    import paper_agent.llm_client as llm_client
    slept = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: slept.append(s))
    return slept
