"""Day 4 packaging: pins, the PDF, the golden set and the notebooks' conventions (cells match the guide)."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks")


def pins(path: Path) -> dict:
    return dict(re.match(r"([a-z0-9-]+)==([^\s;]+)", line).groups() for line in path.read_text().splitlines()
                if re.match(r"[a-z0-9-]+==", line))


def test_requirements_day4_match_the_locks():
    """Colab installs exactly what `make exec-notebooks` executed in Docker; the api image runs the same."""
    day4, runtime, dev = (pins(ROOT / name) for name in ("requirements-day4.txt", "requirements.lock",
                                                          "requirements-dev.lock"))
    expected = {"litellm": "1.104.2", "langchain": "1.4.3", "langchain-core": "1.6.7", "langgraph": "1.2.14",
                "langchain-litellm": "0.11.0", "fastapi": "0.142.4", "uvicorn": "0.54.0", "python-multipart": "0.0.32",
                "httpx": "0.28.1", "pypdf": "6.19.0", "reportlab": "5.0.1", "opentelemetry-sdk": "1.45.1"}
    for package, version in expected.items():
        assert day4[package] == runtime[package] == dev[package] == version, package
    for package, version in day4.items():
        assert dev[package] == version, package
    assert "textstat" not in dev and "opentelemetry-instrumentation-fastapi" not in dev


def test_the_pdf_is_three_pages_deterministic_and_fictional(tmp_path):
    from paper_agent.fixtures.tinycoder_pdf import make_pdf
    from paper_agent.rag import pdf_rag
    data = make_pdf(tmp_path / "t.pdf")
    assert data == make_pdf() and data.startswith(b"%PDF-")
    pages = pdf_rag.parse_pages(tmp_path / "t.pdf")
    text = " ".join(p["text"] for p in pages).lower()
    assert len(pages) == 3 and "fictional" in text and pdf_rag.hidden_text(tmp_path / "t.pdf") == []
    assert not re.search(r"learning rate|fund(ed|ing)|optimi[sz]er|batch size", text)    # the refusal questions
    assert not re.search(r"[\w.]+@[\w-]+\.[a-z]{2,}", text)                                # no email addresses


def codealong() -> dict:
    return json.loads((NOTEBOOKS / "day4" / "Day4_CodeAlong.ipynb").read_text())


def test_codealong_cells_match_the_guide():
    nb = codealong()
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for number in ("0.1", "4.0", "4.1", "4.2", "4.3", "4.3a", "4.4", "4.5", "4.6", "5.1", "5.2", "5.3", "5.3a", "5.4",
                   "5.5", "5.6", "6.1", "6.2", "6.3", "7.1", "7.2", "7.3", "7.4", "7.4a", "7.5", "C.1", "C.2"):
        assert f"### {number}" in text, number
    for number in ("5.0", "6.0", "7.0"):
        assert f"⏩ {number} CATCH-UP" in text, number
    assert "port 8765" in text and "tinycoder_FINAL_v2.pdf" in text and "this-model-does-not-exist" in text
    assert "SEARCH_DELAY_S = 1.5" in text and "EVAL_PAUSE_S" in text and "%%writefile ui.py" in text
    assert "restart the session, then run 4.0" in text


def test_three_todos_each_with_an_offline_test_and_a_rescue():
    cells = codealong()["cells"]
    todos = [i for i, c in enumerate(cells) if "todo" in c["metadata"].get("tags", [])]
    assert len(todos) == 3
    for i, label in zip(todos, ("4.3a", "5.3a", "7.4a"), strict=True):
        test = next(c for c in cells[i + 1:] if c["cell_type"] == "code")
        rescue = next(c for c in cells[cells.index(test) + 1:] if c["cell_type"] == "code")
        assert "todo-test" in test["metadata"]["tags"] and f"# {label} test" in "".join(test["source"])
        assert "rescue" in rescue["metadata"]["tags"]
        assert "".join(rescue["source"]).endswith(cells[i]["metadata"]["solution"])
        body = "".join(test["source"])
        assert "http.post" not in body and "ask_paper" not in body and "run_eval" not in body      # offline


def test_demo_notebook_never_prints_a_full_key():
    nb = json.loads((NOTEBOOKS / "day4" / "Day4_Instructor_Demo.ipynb").read_text())
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    for cell in ("# D0", "# D5.1", "# D5.2", "# D5.6a", "# D5.6b", "# D5.6c", "# D5.6d", "# D6", "# D7", "# DX"):
        assert cell in text, cell
    for line in text.splitlines():
        if "print(" in line and re.search(r"\b[A-Z_]+_KEY\b", line):
            assert "mask(" in line, line
