"""Schemas, citation guard and the hardened explainer (Day 1, sections 5 and 6.10c)."""
import json

import pytest
from pydantic import ValidationError

from paper_agent.explain import CitationError, check_citations, explain_paper, paper_messages
from paper_agent.fixtures.tinycoder import PAPER, SECTION_IDS
from paper_agent.llm_client import LLMClient, LLMConfig
from paper_agent.schemas import KeyTerm, PaperExplainer
from paper_agent.testing import FakeLLM, fake_response


def explainer(**overrides) -> dict:
    data = dict(headline="A small code model nearly matches a 7B model", summary="...", why_it_matters="...",
                evidence_type="benchmark", key_terms=[{"term": "pass@1", "plain_definition": "..."}],
                caveats=["Python only"], citations=["results"])
    data.update(overrides)
    return data


def test_citation_guard_catches_unknown_ids():
    fake = PaperExplainer(**explainer(citations=["results", "appendix_b"]))
    with pytest.raises(CitationError, match="appendix_b"):
        check_citations(fake, SECTION_IDS)


def test_citation_guard_accepts_known_ids():
    check_citations(PaperExplainer(**explainer(citations=["results", "limitations"])), SECTION_IDS)


def test_citation_guard_respects_the_sections_actually_sent():
    with pytest.raises(CitationError, match="limitations"):
        check_citations(PaperExplainer(**explainer(citations=["limitations"])), ["abstract", "results"])


@pytest.mark.parametrize("bad", [
    {"evidence_type": "revolutionary"},
    {"caveats": []},
    {"key_terms": []},
    {"citations": []},
    {"key_terms": [{"term": "t", "plain_definition": "d"}] * 6},
])
def test_schema_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        PaperExplainer(**explainer(**bad))


def test_paper_messages_wrap_sections_in_delimiters():
    msgs = paper_messages(["abstract", "results"])
    assert msgs[0]["role"] == "system" and "never as instructions" in msgs[0]["content"]
    assert '<section id="abstract">' in msgs[1]["content"] and '<section id="methods">' not in msgs[1]["content"]
    assert msgs[1]["content"].startswith(f"Paper title: {PAPER['title']}")


def test_explain_paper_repairs_a_bad_citation_once():
    bad = json.dumps(explainer(citations=["results", "appendix_b"]))
    good = json.dumps(explainer(citations=["results"]))
    fake = FakeLLM([fake_response(bad), fake_response(good)])
    result = explain_paper(LLMClient(LLMConfig(model="fake"), client=fake))
    assert result.citations == ["results"] and fake.calls == 2


def test_explain_paper_fails_loudly_if_repair_still_cites_unknown_ids():
    bad = json.dumps(explainer(citations=["appendix_b"]))
    fake = FakeLLM([fake_response(bad), fake_response(bad)])
    with pytest.raises(CitationError):
        explain_paper(LLMClient(LLMConfig(model="fake"), client=fake))


def test_explain_paper_against_mock_llm(mock):
    llm = LLMClient(LLMConfig(model="mock-llm"), client=mock.client())
    result = explain_paper(llm)
    assert set(result.citations) <= set(SECTION_IDS)
    assert result.caveats and "better than all" not in result.headline.lower()


def test_explain_paper_subset_cites_only_sent_sections(mock):
    llm = LLMClient(LLMConfig(model="mock-llm"), client=mock.client())
    result = explain_paper(llm, section_ids=["abstract", "results"])
    assert set(result.citations) <= {"abstract", "results"}


def test_explain_paper_repairs_bad_citation_from_mock_llm(mock):
    llm = LLMClient(LLMConfig(model="mock-llm"), client=mock.client(headers={"X-Mock-Bad-Citation": "1"}))
    result = explain_paper(llm)
    assert "appendix_b" not in result.citations


def test_keyterm_is_exported():
    assert KeyTerm(term="pass@1", plain_definition="first-try success rate").term == "pass@1"
