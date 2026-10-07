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
    bad = {n: r.detail for n, r in results.items() if r.status not in ("ok", "skipped")}
    assert not bad
    assert "404" in results["unknown model"].detail
    assert cg.recommendations(list(results.values())) == []


def test_check_gateway_reports_max_tokens_quirk(mock):
    cg = load("check_gateway")
    client = mock.client(headers={"X-Mock-Reject-Params": "max_tokens"})
    results = cg.run_checks(client, "mock-llm", None)
    by = {r.name: r.status for r in results}
    assert by["max_tokens"] == "unsupported" and by["max_completion_tokens"] == "ok"
    assert any("max_completion_tokens" in r for r in cg.recommendations(results))


def test_generated_notebooks_match_templates():
    """Generated notebooks must be regenerated (`make notebooks`) after editing notebook_templates/."""
    bn = load("build_notebooks")
    stale = [str(p) for p, builder in bn.targets(bn.TEMPLATE_DAYS)
             if not p.exists() or p.read_text() != bn.render(builder)]
    assert not stale, f"run `make notebooks`: {stale}"


def test_generated_practice_follows_conventions():
    import json
    bn = load("build_notebooks")
    for path, _ in bn.targets(bn.TEMPLATE_DAYS):          # Day 2 is now edited in Colab (see build_notebooks.py)
        if path.name != "Practice.ipynb":
            continue
        cells = json.loads(path.read_text())["cells"]
        tags = [t for c in cells for t in c["metadata"].get("tags", [])]
        assert 1 <= tags.count("todo") <= 4 and tags.count("todo") == tags.count("todo-test") == tags.count("rescue")
        assert tags.count("catch-up") >= 2
        text = "".join("".join(c["source"]) for c in cells)
        assert "Cheatsheet" in text and "Extensions (homework)" in text and "Tomorrow" in text
