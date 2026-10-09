"""Day 4, Sections 6-7: spans, scorers, the judge and the regression gate. Offline: mock-llm and fakes."""
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from paper_agent.evals import gate, metrics, runner
from paper_agent.service import app as service
from paper_agent.service.tracing import setup_tracing, show_trace


def test_one_ask_produces_the_spans_the_slides_name(day4, capsys):
    from paper_agent.service.pipeline import build_agent
    exporter = setup_tracing(InMemorySpanExporter())
    app = service.build_app(build_agent(router=day4.router), day4.store)
    with TestClient(app) as client:
        r = client.post(f"/papers/{day4.paper_id}/ask", headers={"Authorization": "Bearer key-alice"},
                        json={"question": "What attention mechanism does TinyCoder use?"})
    assert r.status_code == 200
    spans = show_trace(exporter)
    names = [s.name for s in spans]
    for name in ("agent.run", "llm.call", "tool.search_paper", "retrieve", "citation_check"):
        assert name in names, name
    assert names[0] == "POST /papers/{paper_id}/ask"            # FastAPI's own request span, no extra package
    llm = [s for s in spans if s.name == "llm.call"]
    assert len(llm) >= 1 and all(s.attributes["llm.input_tokens"] > 0 and s.attributes["llm.cost_usd"] > 0 for s in llm)
    assert llm[0].attributes["llm.deployment"] == "primary"
    assert list(next(s for s in spans if s.name == "retrieve").attributes["retrieve.chunk_ids"])
    assert "agent.run" in capsys.readouterr().out
    exporter.clear()
    setup_tracing(InMemorySpanExporter())                             # stop collecting into this exporter


def test_reading_ease_prefers_short_plain_sentences():
    plain = "The model is small. It runs fast. It works on Python only."
    dense = ("Notwithstanding considerable architectural simplification, the comparatively diminutive transformer "
             "demonstrates competitive functional-correctness characteristics on autoregressive completion.")
    assert metrics.reading_ease(plain) > metrics.reading_ease(dense) + 40
    assert metrics.reading_ease("") == 0.0 and 0.0 <= metrics.reading_ease(dense) <= 100.0


def test_the_gate_lets_wobbles_through_and_catches_real_drops():
    base = {"valid_citations": 1.0, "cited": 1.0, "right_section": 0.83, "readability": 48.0}
    assert gate.regression_gate({"valid_citations": 1.0, "cited": 0.95, "right_section": 0.75, "readability": 40.0},
                                base) == []
    failures = gate.regression_gate({"valid_citations": 0.99, "cited": 0.0, "right_section": 0.83,
                                     "readability": 30.0}, base)
    assert [f.split(":")[0] for f in failures] == ["valid_citations", "cited", "readability"]


def test_an_unparseable_judge_reply_counts_as_not_faithful():
    for reply in ("yes", "", '{"faithful": "true"}', "{not json"):
        assert metrics.judge("q", "sources", "answer", lambda m, t, reply=reply: reply)["faithful"] is False
    seen = {}

    def complete(messages, max_tokens):
        seen.update(max_tokens=max_tokens, system=messages[0]["content"])
        return 'Sure. {"faithful": true, "reason": "supported"}'
    assert metrics.judge("q", "s", "a", complete) == {"faithful": True, "reason": "supported"}
    assert seen == {"max_tokens": 120, "system": metrics.JUDGE_V1}


def test_the_mock_judge_flags_a_planted_claim(day4):
    def complete(messages, max_tokens):
        return day4.router.completion(model="paper-explainer", messages=messages, max_tokens=max_tokens,
                                      temperature=0).choices[0].message.content
    sources = "[p3-c1] TinyCoder solves 41% of functions on the first attempt (pass@1) versus 42% for the 7B baseline."
    good = metrics.judge("pass@1?", sources, "TinyCoder solves 41% of functions on the first attempt [p3-c1].",
                         complete)
    bad = metrics.judge("pass@1?", sources, "TinyCoder solves 41% of functions [p3-c1]. It was trained on ten "
                                            "trillion tokens of Rust from Kaggle notebooks.", complete)
    assert good["faithful"] is True and bad["faithful"] is False


def test_golden_sets_are_marked():
    notebook, everything = runner.golden_set("notebook"), runner.golden_set("all")
    assert len(notebook) == 8 and sum(not q["answerable"] for q in notebook) == 2
    assert len(everything) == 20 and sum(not q["answerable"] for q in everything) == 6
    assert {q["set"] for q in everything} == {"notebook", "extra", "refusal"}
    assert all(q["expected_page"] for q in everything if q["answerable"])


@pytest.fixture
def evaluate(day4):
    from paper_agent.service.pipeline import ask_paper, build_agent

    def complete(messages, max_tokens):
        return day4.router.completion(model="paper-explainer", messages=messages, max_tokens=max_tokens,
                                      temperature=0).choices[0].message.content

    def run(version, golden=None):
        agent = build_agent(version, router=day4.router)
        return runner.run_eval(version, lambda q: ask_paper(day4.paper_id, q, "eval-bot", agent), complete,
                               runner.chunk_index(day4.store.get(day4.paper_id)), golden, show=False)
    return run


def test_v2_is_blocked_on_cited_and_right_section(evaluate):
    v1, _ = evaluate("v1")
    v2, rows = evaluate("v2")
    assert v1["valid_citations"] == 1.0 and v1["cited"] == 1.0 and v1["refused_correctly"] == 1.0
    failures = gate.regression_gate(v2, v1)
    assert sorted(f.split(":")[0] for f in failures) == ["cited", "right_section"]
    assert v2["cited"] == 0.0 and v2["right_section"] == 0.0 and v2["valid_citations"] == 1.0


def test_the_20_question_set_on_the_mock(evaluate):
    m, rows = evaluate("v1", runner.golden_set("all"))
    m = runner.average(rows, runner.ALL_METRICS)
    assert m["valid_citations"] == 1.0 and m["refused_correctly"] == 1.0 and m["cited"] == 1.0
    assert m["recall_at_k"] >= 0.9 and m["page_accuracy"] >= 0.7
