"""Shared fixtures. No API keys: unit tests use fakes; integration tests use mock-llm.

In Docker (`make test`) MOCK_BASE_URL points at the mock-llm service. Run outside Docker,
the mock is started in-process on a free port instead.
"""
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import requests

from paper_agent.llm_client import CALL_LOG, OpenRouterClient

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
        requests.get(f"{url}/health", timeout=5).raise_for_status()
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
        requests.post(f"{self.url}/mock/reset", timeout=5).raise_for_status()

    def fault(self, **fault):
        requests.post(f"{self.url}/mock/faults", json=fault, timeout=5).raise_for_status()

    def stats(self) -> dict:
        return requests.get(f"{self.url}/mock/stats", timeout=5).json()

    def client(self, api_key: str = MOCK_KEY, headers=None, timeout: float = 30.0) -> OpenRouterClient:
        """The course's HTTP client (requests), pointed at the mock."""
        return OpenRouterClient(api_key=api_key, base_url=self.v1, headers=headers, timeout=timeout)

    def openai_client(self, **kwargs):
        """The official OpenAI SDK: only to prove the mock stays OpenAI-compatible."""
        from openai import OpenAI
        kwargs.setdefault("api_key", MOCK_KEY)
        kwargs.setdefault("max_retries", 0)
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


# --- Day 3 ---------------------------------------------------------------------------------------
PDFS = ROOT / "paper_agent" / "fixtures" / "pdfs"
PAPER_FILES = {"tinycoder": "tinycoder_day3.pdf", "tinycoder-injected": "tinycoder_injected.pdf",
               "quickembed": "second_paper.pdf"}


@pytest.fixture(scope="session")
def day3(mock_url, tmp_path_factory):
    """pdf_rag's notebook globals set for mock-llm, a Chroma index in a temporary folder, three papers loaded."""
    from paper_agent.llm_client import LLMClient, LLMConfig
    from paper_agent.rag import pdf_rag as day2

    env = {"LLM_BASE_URL": f"{mock_url}/v1", "LLM_API_KEY": "sk-mock-local", "LLM_MODEL": "mock-llm",
           "EMBED_MODEL": "mock-embed"}
    saved_env = {k: os.environ.get(k) for k in env}
    names = ("llm", "EMBED_MODEL", "collection", "USE_PRECOMPUTED", "PRECOMPUTED")
    saved_globals = {k: getattr(day2, k) for k in names}
    folder, cwd = tmp_path_factory.mktemp("day3"), os.getcwd()
    os.environ.update(env)
    os.chdir(folder)
    day2.llm, day2.EMBED_MODEL, day2.USE_PRECOMPUTED, day2.PRECOMPUTED = LLMClient(LLMConfig()), "mock-embed", False, {}
    day2.collection = day2.open_index("index")
    from paper_agent.agent.papers import load_paper
    for paper_id, name in PAPER_FILES.items():
        load_paper(PDFS / name, paper_id)
    yield day2
    os.chdir(cwd)
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for k, v in saved_globals.items():
        setattr(day2, k, v)
