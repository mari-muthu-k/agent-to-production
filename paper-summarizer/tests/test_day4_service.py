"""Day 4, Section 4: the service. Uploads, auth, ask, the log line, streaming, error statuses. Offline: mock-llm."""
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from paper_agent.service import app as service
from paper_agent.service.config import MAX_UPLOAD_MB
from paper_agent.service.validate import validate_upload

ALICE = {"Authorization": "Bearer key-alice"}
QUESTION = "How much faster is TinyCoder than the 7B baseline?"


@pytest.fixture
def client(day4):
    from paper_agent.service.pipeline import build_agent
    app = service.build_app(build_agent(router=day4.router), day4.store)
    log = service.attach_log_handler(service.LogCapture(echo=False))
    with TestClient(app, raise_server_exceptions=False) as c:
        c.log = log
        yield c
    service.attach_log_handler(__import__("logging").NullHandler())


def test_validate_upload_checks_bytes_and_size():
    with pytest.raises(HTTPException) as e:
        validate_upload(b"This is a plain text file renamed to notes.pdf.")
    assert e.value.status_code == 415
    with pytest.raises(HTTPException) as e:
        validate_upload(b"%PDF-1.4\n" + b"0" * (11 * 1024 * 1024))
    assert e.value.status_code == 413
    validate_upload(b"%PDF-1.4\n" + b"0" * (MAX_UPLOAD_MB * 1024 * 1024 - 100))     # just under: fine


def test_upload_errors_and_auth(client, day4):
    text = client.post("/papers", headers=ALICE, files={"file": ("notes.pdf", b"just text", "application/pdf")})
    big = client.post("/papers", headers=ALICE,
                      files={"file": ("big.pdf", b"%PDF-1.4\n" + b"0" * (11 * 1024 * 1024), "application/pdf")})
    nobody = client.post("/papers", headers={"Authorization": "Bearer key-mallory"},
                         files={"file": ("tinycoder.pdf", day4.pdf, "application/pdf")})
    assert (text.status_code, big.status_code, nobody.status_code) == (415, 413, 401)
    assert text.json() == {"error": "unsupported_type", "message": "Only PDF files are accepted."}
    assert nobody.json()["error"] == "unauthorized" and "Traceback" not in nobody.text


def test_same_bytes_two_names_one_paper(client):
    from conftest import PDFS
    data = (PDFS / "second_paper.pdf").read_bytes()          # not ingested by the fixture: the first upload is new
    first = client.post("/papers", headers=ALICE, files={"file": ("quickembed.pdf", data, "application/pdf")}).json()
    second = client.post("/papers", headers=ALICE,
                         files={"file": ("quickembed_FINAL_v2.pdf", data, "application/pdf")}).json()
    assert first["paper_id"] == second["paper_id"] and len(first["paper_id"]) == 12
    assert (first["status"], second["status"]) == ("processed", "already processed")


def test_ask_returns_usage_and_a_request_id_that_matches_the_log(client, day4):
    r = client.post(f"/papers/{day4.paper_id}/ask", headers=ALICE, json={"question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    for field in ("answer", "citations", "citations_valid", "llm_calls", "input_tokens", "output_tokens",
                  "cost_usd", "answered_by", "latency_ms"):
        assert field in body, field
    assert body["citations_valid"] and body["citations"] and body["answered_by"] == ["primary"]
    assert body["llm_calls"] == 2 and body["cost_usd"] > 0
    line = client.log.find(r.headers["X-Request-ID"])
    assert line["user"] == "alice" and line["status"] == 200 and line["llm_calls"] == 2
    assert line["route"] == f"POST /papers/{day4.paper_id}/ask"
    assert set(line) == {"request_id", "user", "route", "status", "latency_ms", "llm_calls", "input_tokens",
                         "output_tokens", "cost_usd", "answered_by"}
    logged = json.dumps(line)
    assert QUESTION not in logged and body["answer"][:40] not in logged and "TinyCoder" not in logged


def test_every_response_has_a_request_id(client):
    for r in (client.get("/health"), client.get("/me/usage"), client.get("/me/usage", headers=ALICE)):
        assert len(r.headers["X-Request-ID"]) == 8


def test_explain_streams_progress_then_answer_then_done(client, day4):
    with client.stream("POST", f"/papers/{day4.paper_id}/explain", headers=ALICE) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line.removeprefix("event: ") for line in r.iter_lines() if line.startswith("event: ")]
    first_other = next(i for i, e in enumerate(events) if e != "progress")
    assert events[0] == "progress" and first_other >= 3
    assert events[first_other:] == ["answer", "done"]


def test_unknown_paper_is_404(client):
    r = client.post("/papers/000000000000/ask", headers=ALICE, json={"question": "Hi?"})
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_provider_down_is_503_without_a_stack_trace(client, day4, mock):
    from paper_agent.service.gateway import make_router
    from paper_agent.service.pipeline import build_agent
    client.app.state.agent = build_agent(router=make_router(primary="broken-model", fallback=False, num_retries=0))
    r = client.post(f"/papers/{day4.paper_id}/ask", headers=ALICE, json={"question": QUESTION})
    assert r.status_code == 503 and r.json()["error"] == "provider_unavailable"
    assert "Traceback" not in r.text and "litellm" not in r.text.lower()


def test_an_unexpected_error_is_a_500_without_a_stack_trace(client, day4, monkeypatch):
    monkeypatch.setattr(service, "ask_paper", lambda *a, **k: 1 / 0)
    r = client.post(f"/papers/{day4.paper_id}/ask", headers=ALICE, json={"question": QUESTION})
    assert r.status_code == 500 and r.json() == {"error": "internal_error",
                                                 "message": "Something went wrong on our side."}
    assert "ZeroDivisionError" not in r.text and r.headers["X-Request-ID"]
