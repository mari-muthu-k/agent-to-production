"""Day 2 ingestion: loaders, cleaning, splitters, embedders, stores, sample papers. Offline."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import openai
import pytest

from paper_agent.ingest.cleaning import Block, heading, to_blocks
from paper_agent.ingest.embeddings import EmbeddingMismatchError, HashingEmbedder, OpenAIEmbedder
from paper_agent.ingest.loaders import load_pdf
from paper_agent.ingest.pipeline import ingest_pdf
from paper_agent.ingest.splitters import chunk_stats, fixed_chunks, semantic_chunks, sentence_chunks
from paper_agent.ingest.store import ChromaStore, NumpyStore, collection_name, top_k
from paper_agent.llm_client import CALL_LOG
from paper_agent.papers import edge_case_pdf, fetch_paper, paper_ids, registry

ROOT = Path(__file__).resolve().parents[1]
EMB = HashingEmbedder()


@pytest.fixture(scope="module")
def mistral_blocks():
    return to_blocks(load_pdf(fetch_paper("mistral-7b")).pages)


# --- sample papers -------------------------------------------------------------
def test_registry_papers_are_local_licensed_and_verified():
    assert len(paper_ids()) >= 3
    for p in registry()["papers"]:
        assert p["license"] == "CC BY 4.0" and p["authors"] and p["arxiv"]
        path = fetch_paper(p["id"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == p["sha256"]
    manifest = (ROOT / "data" / "papers" / "MANIFEST.md").read_text()
    assert all(p["arxiv"] in manifest for p in registry()["papers"])


def test_edge_case_pdfs_are_reproducible(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mk", ROOT / "tools" / "make_edge_case_pdfs.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    monkeypatch.setattr(mk, "OUT", tmp_path)
    mk.main()
    for name in ("two_column", "image_only", "hidden_white", "hidden_tiny"):
        assert (tmp_path / f"{name}.pdf").read_bytes() == edge_case_pdf(name).read_bytes(), name


# --- loaders -------------------------------------------------------------------
def test_naive_extraction_interleaves_two_columns():
    naive = load_pdf(edge_case_pdf("two_column"), layout="naive").pages[0].text
    fixed = load_pdf(edge_case_pdf("two_column")).pages[0]
    assert "Abstract 3 Results" in naive                 # left and right headings on one line
    assert fixed.columns == 2 and "Abstract 3 Results" not in fixed.text
    assert fixed.text.index("2 Method") < fixed.text.index("3 Results")   # left column first


def test_image_only_page_is_flagged():
    doc = load_pdf(edge_case_pdf("image_only"))
    assert doc.pages[1].needs_ocr and not doc.pages[1].text
    assert doc.warnings == ["page 2: no text layer (image-only); needs OCR, skipped"]


def test_real_paper_words_are_not_glued():
    text = load_pdf(fetch_paper("mistral-7b")).pages[0].text
    assert "Mistral 7B" in text and "Mistral7B" not in text


# --- cleaning ------------------------------------------------------------------
@pytest.mark.parametrize("line,expected", [
    ("Abstract", "abstract"), ("1 Introduction", "introduction"), ("3.2 Ablation Studies", "ablation-studies"),
    ("2 Architectural details", "architectural-details"), ("A Appendix Details", "appendix-details"),
    ("We train the model.", None), ("Block 1 Years ago our fathers b", None), ("Table 3: Results", None),
    ("2023", None), ("B C", None),
])
def test_heading_detection(line, expected):
    assert heading(line) == expected


def test_blocks_track_section_and_page(mistral_blocks):
    by_section = {}
    for b in mistral_blocks:
        by_section.setdefault(b.section, b.page)               # first page of each section
    assert by_section["abstract"] == 1 and by_section["architectural-details"] == 2
    assert "references" not in by_section and all(b.text for b in mistral_blocks)


# --- splitters -----------------------------------------------------------------
def test_fixed_chunks_cut_mid_sentence(mistral_blocks):
    stats = chunk_stats(fixed_chunks(mistral_blocks, "m", size=300))
    assert stats["max_chars"] == 300 and stats["cut_mid_sentence"] > stats["chunks"] / 2


def test_sentence_chunks_respect_size_and_sentences(mistral_blocks):
    chunks = sentence_chunks(mistral_blocks, "m")
    stats = chunk_stats(chunks)
    assert stats["max_chars"] <= 1000 and 500 <= stats["median_chars"] <= 1000
    assert stats["cut_mid_sentence"] <= len(chunks) * 0.25
    assert all(c.page in c.pages and c.id.startswith("m:") for c in chunks)


def test_sentence_chunks_overlap():
    text = " ".join(f"Sentence number {i} talks about attention heads and caches." for i in range(60))
    chunks = sentence_chunks([Block("results", 4, text)], "p", chunk_size=300, chunk_overlap=100)
    assert len(chunks) > 3
    assert any(chunks[i].text[-40:] in chunks[i + 1].text for i in range(len(chunks) - 1))


def test_chunk_pages_span_page_breaks():
    blocks = [Block("methods", 2, "A" * 50 + ". " + "Alpha beta gamma. " * 30), Block("methods", 3, "Delta. " * 40)]
    chunks = sentence_chunks(blocks, "p", chunk_size=600, chunk_overlap=0)
    assert chunks[0].page == 2 and any(3 in c.pages for c in chunks)


def test_semantic_chunks_within_bounds(mistral_blocks):
    chunks = semantic_chunks(mistral_blocks, "m", EMB.embed)
    long_sections = [c for c in chunks if len(c.text) > 1000]
    assert chunks and not long_sections
    assert sum(1 for c in chunks if len(c.text) >= 400) >= len(chunks) * 0.6


# --- embedders -----------------------------------------------------------------
def test_hashing_embedder_is_normalised_and_lexical():
    v = EMB.embed(["sliding window attention", "attention over a sliding window", "baking bread at home"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > 0.5 > v[0] @ v[2]


def test_different_models_live_in_different_spaces():
    a, b = HashingEmbedder(salt="model-a"), HashingEmbedder(salt="model-b")
    text = ["grouped-query attention speeds up inference"]
    assert float(a.embed(text)[0] @ a.embed(text)[0]) > 0.99
    assert abs(float(a.embed(text)[0] @ b.embed(text)[0])) < 0.3


def test_openai_embedder_against_mock(mock):
    emb = OpenAIEmbedder("mock-embed", client=mock.client(), batch_size=2)
    v = emb.embed(["a", "b", "c", "a"])
    assert v.shape == (4, 256) and np.allclose(v[0], v[3])
    assert mock.stats()["embedding_requests"] == 2            # 3 unique texts, batches of 2
    assert emb.tokens_used > 0 and CALL_LOG[-1].model == "mock-embed"
    emb.embed(["a"])
    assert mock.stats()["embedding_requests"] == 2            # cached


def test_openai_embedder_retries_429(mock, no_sleep):
    mock.fault(status=429, retry_after=2, count=1, endpoint="embeddings")
    emb = OpenAIEmbedder("mock-embed", client=mock.client())
    assert emb.embed(["pass@1"]).shape == (1, 256) and no_sleep == [2.0]


def test_openai_embedder_does_not_retry_400(mock, no_sleep):
    emb = OpenAIEmbedder("no-such-embedding-model", client=mock.client())
    with pytest.raises(openai.NotFoundError):
        emb.embed(["x"])
    assert no_sleep == []


# --- stores --------------------------------------------------------------------
def test_top_k_orders_by_cosine():
    m = np.eye(4, dtype=np.float32)
    q = np.array([0.1, 0.9, 0.3, 0.0], dtype=np.float32)
    assert [i for i, _ in top_k(m, q, 2)] == [1, 2]


@pytest.fixture
def small_index(mistral_blocks):
    chunks = sentence_chunks(mistral_blocks, "mistral-7b")
    return chunks, EMB.embed([c.text for c in chunks])


def test_numpy_store_search_save_load(small_index, tmp_path):
    chunks, vecs = small_index
    store = NumpyStore(EMB.model)
    store.add(chunks, vecs, EMB.model)
    store.add(chunks, vecs, EMB.model)                        # idempotent
    assert len(store) == len(chunks)
    hits = store.search(EMB.embed(["rolling buffer cache"])[0], k=3, embed_model=EMB.model)
    assert hits[0].score >= hits[1].score and hits[0].chunk.section == "architectural-details"
    store.save(tmp_path / "idx")
    again = NumpyStore.load(tmp_path / "idx")
    assert [h.chunk.id for h in again.search(EMB.embed(["rolling buffer cache"])[0], k=3)] == \
           [h.chunk.id for h in hits]


def test_numpy_store_rejects_other_models(small_index):
    chunks, vecs = small_index
    store = NumpyStore(EMB.model)
    store.add(chunks, vecs)
    with pytest.raises(EmbeddingMismatchError):
        store.search(vecs[0], embed_model="text-embedding-3-small")
    with pytest.raises(EmbeddingMismatchError):
        store.search(np.ones(64, dtype=np.float32))


def test_chroma_store_matches_numpy(small_index, tmp_path):
    chunks, vecs = small_index
    np_store, ch_store = NumpyStore(EMB.model), ChromaStore(EMB.model, path=tmp_path)
    np_store.add(chunks, vecs)
    ch_store.add(chunks, vecs)
    ch_store.add(chunks, vecs)                                # upsert: no duplicates
    assert len(ch_store) == len(chunks)
    q = EMB.embed(["How does the rolling buffer cache limit memory use?"])[0]
    a, b = np_store.search(q, k=5), ch_store.search(q, k=5)
    assert [h.chunk.id for h in a] == [h.chunk.id for h in b]
    assert np.allclose([h.score for h in a], [h.score for h in b], atol=1e-4)
    assert ch_store.search(q, k=5, paper_id="other") == []
    persisted = ChromaStore(EMB.model, path=tmp_path)         # reopen from disk
    assert len(persisted) == len(chunks)


def test_chroma_collection_per_model(tmp_path):
    assert collection_name("text-embedding-3-small") != collection_name("mock-embed")
    store = ChromaStore("model-a", path=tmp_path, collection="shared")
    store.add([], np.zeros((0, 4)))
    with pytest.raises(EmbeddingMismatchError):
        ChromaStore("model-b", path=tmp_path, collection="shared")


def test_ingest_pdf_report():
    store = NumpyStore(EMB.model)
    report, chunks = ingest_pdf(edge_case_pdf("image_only"), EMB, store, paper_id="chunkbench")
    assert report.pages == 2 and report.chunks == len(chunks) == len(store) > 0
    assert "needs OCR" in str(report)
