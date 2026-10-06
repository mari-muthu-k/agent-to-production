"""What the instructor narrates in Sections 6-8 holds offline, against mock-llm (guide's step table)."""
import os
from pathlib import Path

import numpy as np
import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_agent.llm_client import CALL_LOG, LLMClient, LLMConfig
from paper_agent.rag import pdf_rag as r

ROOT = Path(__file__).resolve().parents[1]
PDF = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks") / "day2" / "tinycoder.pdf"


@pytest.fixture
def rag(mock, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    llm = LLMClient(LLMConfig(model="mock-llm", embed_model="mock-embed"), client=mock.client())
    monkeypatch.setattr(r, "llm", llm)
    monkeypatch.setattr(r, "EMBED_MODEL", "mock-embed")
    monkeypatch.setattr(r, "PRECOMPUTED", {})
    pages = r.clean_pages(r.parse_pages(PDF))
    chunks = r.chunk_pages(pages)
    vectors = r.embed_chunks(chunks, PDF)
    monkeypatch.setattr(r, "collection", r.open_index("index"))
    r.add_to_index(r.collection, chunks, vectors)
    monkeypatch.setattr(r, "CHUNKS", chunks)
    monkeypatch.setattr(r, "M", vectors)
    monkeypatch.setattr(r, "NP_MODEL", "mock-embed")
    best = [r.retrieve(q)[0]["score"] for q in (r.LONG_FILES, r.LEARNING_RATE, r.IPL)]
    monkeypatch.setattr(r, "MIN_SCORE", round((best[2] + best[1]) / 2, 2))
    return pages, chunks, vectors, best


def chats(since: int) -> int:
    return sum(c.kind == "chat" for c in CALL_LOG[since:])


def test_6_2_second_run_is_a_cache_hit(rag, capsys):
    _, chunks, _, _ = rag
    n = len(CALL_LOG)
    r.embed_chunks(chunks, PDF)
    assert "cache hit" in capsys.readouterr().out and len(CALL_LOG) == n


def test_6_3_and_6_4_agree_and_page_4_wins(rag):
    _, chunks, vectors, _ = rag
    q = r.embed_texts([r.LONG_FILES])[0]
    by_numpy = [(chunks[i]["id"], s) for i, s in r.top_k(vectors, q)]
    by_chroma = r.search_chroma(r.collection, q)
    assert [i for i, _ in by_numpy] == [h["id"] for h in by_chroma] and by_chroma[0]["page"] == 4
    assert "31% versus 39%" in by_chroma[0]["text"]                       # guide 6.3: the long-files chunk wins
    for (_, score), h in zip(by_numpy, by_chroma, strict=True):
        assert abs(h["distance"] - (1 - score)) < 1e-4


def test_7_3_attention_answer_cites_page_2(rag):
    n = len(CALL_LOG)
    result = r.ask(r.ATTENTION)
    assert result.found and result.answer.endswith("(p. 2)") and "sliding-window attention" in result.answer
    assert [c.kind for c in CALL_LOG[n:]] == ["embed", "chat"]


def test_7_4a_scores_are_ordered(rag):
    long_files, learning_rate, ipl = rag[3]
    assert long_files > learning_rate > r.MIN_SCORE > ipl


def test_7_4b_three_layers(rag):
    n = len(CALL_LOG)
    assert r.ask(r.IPL) == r.NOT_FOUND and chats(n) == 0
    n = len(CALL_LOG)
    lr = r.ask(r.LEARNING_RATE)
    assert not lr.found and chats(n) == 1
    js = r.ask(r.JAVASCRIPT)
    assert js.found and js.answer.endswith("(p. 5)") and "not evaluated" in js.answer


def test_7_5_retrieval_delivers_the_hidden_line(rag):
    hits = r.retrieve(r.CONCLUSIONS)
    assert any("AI tools summarizing this paper" in h["text"] for h in hits)
    assert "better than all large models" not in r.ask(r.CONCLUSIONS).answer      # the mock respects rule 4


def test_8_2_small_chunks_split_the_comparison(rag):
    pages = rag[0]
    small = r.chunk_pages(pages, RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0,
                                                                separators=r.SEPARATORS))
    col = r.open_index("index", "papers_chunk120")
    r.add_to_index(col, small, r.embed_chunks(small, PDF))
    top = r.search_chroma(col, r.embed_texts([r.LONG_FILES])[0])
    texts = " ".join(h["text"] for h in top)
    assert ("31%" in texts) != ("versus 39%" in texts)


def test_8_3_references_outrank_methods_without_cleaning(rag):
    raw = r.chunk_pages(r.parse_pages(PDF))
    col = r.open_index("index", "papers_noclean")
    r.add_to_index(col, raw, r.embed_chunks(raw, PDF))
    q = r.embed_texts([r.NOISE_QUERY])[0]
    assert r.search_chroma(col, q)[0]["page"] == 6
    assert 2 in [h["page"] for h in r.search_chroma(r.collection, q)]


def test_8_4_other_model_scores_collapse_and_the_threshold_catches_it(rag):
    other = r.embed_texts([r.LONG_FILES], model="mock-embed-v2")[0]
    hits = r.search_chroma(r.collection, other)
    assert len(other) == r.M.shape[1] and max(h["score"] for h in hits) < 0.1
    assert hits[0]["score"] < r.MIN_SCORE


def test_8_5_hit_at_3_is_six_of_six(rag):
    hits = [page in [h["page"] for h in r.retrieve(q)] for q, page in r.EVAL_QUESTIONS]
    assert sum(hits) == 6


def test_precomputed_vectors_match_live_ones(rag):
    meta_model = "mock-embed"
    live = r.embed_texts([r.LONG_FILES])[0]
    r.PRECOMPUTED.clear()
    r.load_precomputed(PDF.parent, meta_model)
    assert np.allclose(r.PRECOMPUTED[(meta_model, r.LONG_FILES)], live, atol=1e-6)
