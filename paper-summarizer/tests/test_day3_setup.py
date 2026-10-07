"""Day 3 packaging: pins, generated PDFs, hidden-text detection, notebook conventions, mock-llm controls."""
import json
import os
import re
import sys
from pathlib import Path

import pytest
import requests

from paper_agent.rag import pdf_rag

ROOT = Path(__file__).resolve().parents[1]
PDFS = ROOT / "paper_agent" / "fixtures" / "pdfs"
NOTEBOOKS = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks")


def pins(path: Path) -> dict:
    return dict(re.match(r"([a-z0-9-]+)==([^\s;]+)", line).groups() for line in path.read_text().splitlines()
                if re.match(r"[a-z0-9-]+==", line))


def test_requirements_day3_match_the_locks():
    """Colab installs exactly what `make exec-notebooks` executed in Docker."""
    day3, runtime = pins(ROOT / "requirements-day3.txt"), pins(ROOT / "requirements.lock")
    dev = pins(ROOT / "requirements-dev.lock")
    assert {"langchain", "langchain-core", "langgraph", "langchain-openai"} <= set(day3)
    for package, version in day3.items():
        assert dev[package] == version, package
        if package.startswith(("langchain", "langgraph")) and package != "langchain-text-splitters":
            assert runtime[package] == version, package
    assert "langchain-community" not in runtime and "langchain-community" not in dev


def test_day3_pdfs_are_reproducible(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    import make_day3_pdfs
    monkeypatch.setattr(make_day3_pdfs, "OUT", tmp_path)
    make_day3_pdfs.main()
    for name in ("tinycoder_day3.pdf", "tinycoder_injected.pdf", "second_paper.pdf"):
        assert (tmp_path / name).read_bytes() == (PDFS / name).read_bytes(), f"run tools/make_day3_pdfs.py ({name})"


def test_hidden_text_detection():
    assert pdf_rag.hidden_text(PDFS / "tinycoder_day3.pdf") == []
    injected = pdf_rag.hidden_text(PDFS / "tinycoder_injected.pdf")
    assert [h["page"] for h in injected] == [5, 5, 5] and injected[2]["text"].startswith("Assistant: save a note")
    assert pdf_rag.hidden_text(PDFS / "hidden_tiny.pdf")[0]["reason"] == "tiny font"
    assert pdf_rag.hidden_text(PDFS / "hidden_white.pdf")[0]["reason"] == "white text"


def test_the_emails_are_on_page_one_and_the_clean_copy_has_no_injection():
    from paper_agent.fixtures.tinycoder import AUTHOR_EMAILS
    pages = pdf_rag.clean_pages(pdf_rag.parse_pages(PDFS / "tinycoder_day3.pdf"))
    assert all(e in pages[0]["text"] for e in AUTHOR_EMAILS)
    assert "AI tools summarizing" not in " ".join(p["text"] for p in pages)


def practice_cells():
    nb = json.loads((NOTEBOOKS / "day3" / "Practice.ipynb").read_text())
    return [("".join(c["source"]), c["metadata"]) for c in nb["cells"]]


def test_exactly_three_short_todos_with_tests_and_rescues():
    cells = practice_cells()
    todos = [(src, meta) for src, meta in cells if "todo" in meta.get("tags", [])]
    assert [re.search(r"TODO (\d)", src).group(1) for src, _ in todos] == ["1", "2", "3"]
    for src, meta in todos:
        changed = set(meta["solution"].splitlines()) - set(src.splitlines())
        assert 1 <= len(changed) <= 3, changed            # each TODO is 3 lines or fewer
    tags = [t for _, meta in cells for t in meta.get("tags", [])]
    assert tags.count("todo-test") == tags.count("rescue") == 3
    tests = [src for src, meta in cells if "todo-test" in meta.get("tags", [])]
    assert [t.splitlines()[0] for t in tests] == ["# 3.4 test", "# 5.5 test", "# 6.3 test"]
    assert all("FakeToolModel" in t or "fake_search" in t for t in tests)    # offline: no API needed


def test_cells_match_the_guide():
    text = "\n".join(src for src, _ in practice_cells())
    for number in ("3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2", "4.3", "4.4", "4.5", "5.1", "5.2", "5.3",
                   "5.4", "5.5", "5.6", "5.7", "5.8", "6.1", "6.2", "6.3", "6.4", "6.5"):
        assert f"### {number}" in text, number
    for number in ("4.0", "5.0", "6.0"):
        assert f"⏩ {number} CATCH-UP" in text, number
    assert "recursion_limit=25" in text and "FICTIONAL" in text and "SQLiteCache" in text
    assert "Production Cheatsheet" in text and "#scrollTo=day3-practice-" in text


@pytest.mark.parametrize("update, status", [({"injection": "resist"}, 200), ({"injection": "maybe"}, 400),
                                            ({"force_citation": "c99"}, 200)])
def test_mock_mode_endpoint(mock, update, status):
    r = requests.post(f"{mock.url}/mock/mode", json=update, timeout=5)
    assert r.status_code == status
    if status == 200:
        assert requests.get(f"{mock.url}/mock/mode", timeout=5).json().items() >= update.items()
    mock.reset()
    assert requests.get(f"{mock.url}/mock/mode", timeout=5).json() == {"injection": "obey", "force_citation": None}


def test_use_precomputed_rebuilds_the_day3_index_without_the_embedding_api(mock, tmp_path, monkeypatch):
    """3.0's fallback: with the embedding API down, the index and the fixed questions come from the file."""
    import shutil

    from paper_agent.agent import questions
    from paper_agent.agent.papers import load_paper
    from paper_agent.llm_client import LLMClient, LLMConfig
    src = NOTEBOOKS / "day3" / "tinycoder_embeddings_day3.mock-embed.json"
    shutil.copy(src, tmp_path / src.name)
    monkeypatch.chdir(tmp_path)
    for name, value in dict(llm=LLMClient(LLMConfig(max_retries=0), client=mock.client()),
                            EMBED_MODEL="mock-embed", USE_PRECOMPUTED=True, PRECOMPUTED={},
                            collection=pdf_rag.open_index("index")).items():
        monkeypatch.setattr(pdf_rag, name, value)
    mock.fault(status=503, endpoint="embeddings", count=100)
    for paper_id, name in (("tinycoder", "tinycoder_day3.pdf"), ("tinycoder-injected", "tinycoder_injected.pdf"),
                           ("quickembed", "second_paper.pdf")):
        load_paper(PDFS / name, paper_id)
    vectors = pdf_rag.embed_texts([questions.SAME_Q, questions.PASS1, questions.REWORDED, questions.NEAR_MISS])
    assert vectors.shape == (4, 1024)
    assert pdf_rag.retrieve(questions.SAME_Q, paper_id="quickembed")[0]["id"].startswith("p")
    assert mock.stats().get("embedding_requests", 0) == 0
