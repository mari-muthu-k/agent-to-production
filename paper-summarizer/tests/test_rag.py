"""Day 2 query side: threshold, grounded answers, page-citation guard, number check, evals."""
import json

import pytest

from paper_agent.ingest.embeddings import HashingEmbedder, OpenRouterEmbedder
from paper_agent.ingest.pipeline import ingest_pdf
from paper_agent.ingest.splitters import Chunk
from paper_agent.ingest.store import Hit, NumpyStore
from paper_agent.llm_client import BadRequestError, LLMClient, LLMConfig, OpenRouterClient
from paper_agent.papers import fetch_paper, paper_ids
from paper_agent.rag.answer import (
    PageCitationError,
    allowed_citations,
    answer_question,
    check_page_citations,
    fit_to_budget,
    rag_messages,
    unsupported_numbers,
)
from paper_agent.rag.evals import golden_questions, hit_rate_at_k
from paper_agent.rag.naive import naive_answer, naive_chunks, naive_index, naive_search
from paper_agent.rag.retrieve import Retriever
from paper_agent.schemas import GroundedAnswer
from paper_agent.testing import FakeLLM, fake_response

CHUNK = Chunk("p:0001", "p", "On 5,000 requests the draft tokens were accepted 71% of the time.", "results", 4, [4])
OTHER = Chunk("p:0002", "p", "We used a 68M-parameter draft model.", "method", 2, [2, 3])
HITS = [Hit(CHUNK, 0.8), Hit(OTHER, 0.5)]


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k=5, paper_id=None, min_score=None):
        return [h for h in self.hits if min_score is None or h.score >= min_score]


def grounded(**kw) -> str:
    data = {"status": "answered", "answer": "71% of draft tokens were accepted.",
            "citations": [{"section": "results", "page": 4}], "confidence": "high"}
    data.update(kw)
    return json.dumps(data)


# --- schema --------------------------------------------------------------------
def test_answered_requires_citations():
    with pytest.raises(ValueError):
        GroundedAnswer(status="answered", answer="x", citations=[], confidence="high")
    GroundedAnswer(status="insufficient_evidence", answer="not covered", citations=[], confidence="low")


# --- citation guard and checks -------------------------------------------------
def test_allowed_citations_cover_every_page_of_a_chunk():
    assert allowed_citations(HITS) == {("results", 4), ("method", 2), ("method", 3)}


def test_page_citation_guard():
    ok = GroundedAnswer.model_validate_json(grounded())
    check_page_citations(ok, HITS)
    wrong_page = GroundedAnswer.model_validate_json(grounded(citations=[{"section": "results", "page": 9}]))
    with pytest.raises(PageCitationError, match="results.*9"):
        check_page_citations(wrong_page, HITS)


def test_unsupported_numbers():
    good = GroundedAnswer.model_validate_json(grounded())
    assert unsupported_numbers(good, HITS) == []
    bad = GroundedAnswer.model_validate_json(grounded(answer="71% accepted on 512 A100 GPUs."))
    assert unsupported_numbers(bad, HITS) == ["512", "100"]


def test_fit_to_budget_keeps_best_first():
    assert [h.chunk.id for h in fit_to_budget(HITS, 40)] == ["p:0001"]


def test_rag_messages_wrap_chunks_with_section_and_page():
    msgs = rag_messages("What was accepted?", HITS)
    assert '<chunk id="p:0001" section="results" page="4">' in msgs[1]["content"]
    assert "never as instructions" in msgs[0]["content"] and "insufficient_evidence" in msgs[0]["content"]


# --- answer_question with fakes ------------------------------------------------
def llm(script) -> tuple:
    fake = FakeLLM(script)
    return LLMClient(LLMConfig(model="fake"), client=fake), fake


def test_no_evidence_means_no_llm_call():
    client, fake = llm([])
    out = answer_question("What GPU?", FixedRetriever(HITS), client, min_score=0.9)
    assert out.status == "insufficient_evidence" and not out.llm_called and fake.calls == 0


def test_answered_with_page_citations():
    client, fake = llm([fake_response(grounded())])
    out = answer_question("How often were draft tokens accepted?", FixedRetriever(HITS), client)
    assert out.status == "answered" and out.citations[0].page == 4 and not out.warnings and fake.calls == 1


def test_bad_citation_is_repaired_once():
    client, fake = llm([fake_response(grounded(citations=[{"section": "appendix", "page": 12}])),
                        fake_response(grounded())])
    out = answer_question("q", FixedRetriever(HITS), client)
    assert out.citations[0].section == "results" and fake.calls == 2


def test_bad_citation_twice_fails_loudly():
    bad = grounded(citations=[{"section": "appendix", "page": 12}])
    client, _ = llm([fake_response(bad), fake_response(bad)])
    with pytest.raises(PageCitationError):
        answer_question("q", FixedRetriever(HITS), client)


def test_number_warning():
    client, _ = llm([fake_response(grounded(answer="Accepted 71% of the time on 512 GPUs."))])
    out = answer_question("q", FixedRetriever(HITS), client)
    assert out.warnings and "512" in out.warnings[0]


# --- against mock-llm ----------------------------------------------------------
@pytest.fixture(scope="module")
def mock_index(mock_url):
    client = OpenRouterClient(api_key="sk-mock-local", base_url=f"{mock_url}/v1")
    emb = OpenRouterEmbedder("mock-embed", client=client)
    store = NumpyStore(emb.model)
    for pid in paper_ids():
        ingest_pdf(fetch_paper(pid), emb, store, paper_id=pid)
    return Retriever(store, emb), client


def test_retrieval_eval_baseline(mock_index):
    retriever, _ = mock_index
    report = hit_rate_at_k(retriever, golden_questions(), k=5)
    assert len(report.rows) == 10 and report.hit_rate >= 0.9      # regression guard on the mock


def test_threshold_separates_on_and_off_topic(mock_index):
    retriever, _ = mock_index
    assert retriever.retrieve("How does the rolling buffer cache limit memory use?")
    assert retriever.retrieve("What is the best pizza topping for a party?") == []


def test_end_to_end_answer_and_idk(mock_index):
    retriever, client = mock_index
    model = LLMClient(LLMConfig(model="mock-llm"), client=client)
    ok = answer_question("How does the rolling buffer cache limit memory use?", retriever, model)
    assert ok.status == "answered" and ok.citations
    assert {(c.section, c.page) for c in ok.citations} <= allowed_citations(ok.hits)
    idk = answer_question("Which GPU cloud provider sponsored the Mistral 7B training run?", retriever, model,
                          min_score=0.0)
    assert idk.status == "insufficient_evidence" and idk.llm_called


def test_naive_rag_hallucinates_without_an_idk_rule(mock_index):
    _, client = mock_index
    text = load_text()
    chunks = naive_chunks(text)
    emb = HashingEmbedder()
    hits = naive_search("Which GPU cloud provider sponsored the training run?", chunks, naive_index(chunks, emb.embed),
                        emb.embed)
    reply = naive_answer("Which GPU cloud provider sponsored the training run?", [c for _, c in hits], client,
                         "mock-llm")
    assert "A100" in reply                                        # confidently made up


def test_mock_context_window_overflow(mock_index):
    _, client = mock_index
    huge = load_text() * 8
    with pytest.raises(BadRequestError, match="maximum context length"):
        client.chat(model="mock-llm", max_tokens=50,
                                       messages=[{"role": "user", "content": f"{huge}\n\nQuestion: why?"}])


def load_text() -> str:
    from paper_agent.ingest.loaders import load_pdf
    return load_pdf(fetch_paper("mistral-7b")).text
