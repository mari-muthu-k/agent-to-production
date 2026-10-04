"""tools/check_gateway.py against mock-llm: every probe passes, and quirks are reported."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_gateway_all_ok_against_mock(mock):
    cg = load("check_gateway")
    results = {r.name: r for r in cg.run_checks(mock.client(), "mock-llm", "mock-embed")}
    bad = {n: r.detail for n, r in results.items() if r.status != "ok"}
    assert not bad
    assert "404" in results["unknown model"].detail
    assert cg.recommendations(list(results.values())) == []


def test_check_gateway_reports_max_tokens_quirk(mock):
    cg = load("check_gateway")
    client = mock.client(default_headers={"X-Mock-Reject-Params": "max_tokens"})
    results = cg.run_checks(client, "mock-llm", None)
    by = {r.name: r.status for r in results}
    assert by["max_tokens"] == "unsupported" and by["max_completion_tokens"] == "ok"
    assert any("max_completion_tokens" in r for r in cg.recommendations(results))


def test_build_notebooks_verifies_day1_checksums():
    assert load("build_notebooks").build_day1(None) == 0
