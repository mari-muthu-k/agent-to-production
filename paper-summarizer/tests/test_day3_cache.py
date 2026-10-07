"""Day 3, Section 6: the four caches, against mock-llm."""
from types import SimpleNamespace

import requests
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AIMessage

from paper_agent.agent import questions as Q
from paper_agent.agent.guards import CITATION_FAILED, answer_passed_guards
from paper_agent.agent.model import make_chat_model
from paper_agent.agent.prompts import AGENT_SYSTEM_V1
from paper_agent.agent.tools import section_text
from paper_agent.cache.keys import make_key, normalize
from paper_agent.cache.prompt_cache import cached_input_tokens
from paper_agent.cache.semantic import SemanticCache
from paper_agent.cache.tool_cache import cached_search, naive_key, search_key


def chat_requests(mock) -> int:
    return requests.get(f"{mock.url}/mock/stats", timeout=5).json().get("chat_requests", 0)


def test_exact_cache_hit_sends_no_request(day3, mock):
    model = make_chat_model(cache=InMemoryCache())
    prompt = [{"role": "user", "content": section_text("results", "tinycoder") + "\n\nSummarize the results section."}]
    first = model.invoke(prompt)
    n = chat_requests(mock)
    second = model.invoke(prompt)
    assert chat_requests(mock) == n and second.content == first.content
    assert second.usage_metadata["total_cost"] == 0 and "total_cost" not in first.usage_metadata
    model.invoke([{**prompt[0], "content": prompt[0]["content"].replace("Summarize", "Summarise")}])
    assert chat_requests(mock) == n + 1                            # one letter changed: a miss


def test_naive_key_serves_the_wrong_paper(day3):
    cache = {}
    tiny, _ = cached_search(Q.SAME_Q, "tinycoder", naive_key, cache=cache)
    other, hit = cached_search(Q.SAME_Q, "quickembed", naive_key, cache=cache)
    assert hit and other == tiny and "TinyCoder" in other[0]["text"]


def test_correct_key_separates_papers_and_hits_on_repeat(day3):
    cache = {}
    tiny, hit1 = cached_search(Q.SAME_Q, "tinycoder", search_key, cache=cache)
    quick, hit2 = cached_search(Q.SAME_Q, "quickembed", search_key, cache=cache)
    again, hit3 = cached_search("  what is the MAIN result ", "quickembed", search_key, cache=cache)
    assert (hit1, hit2, hit3) == (False, False, True)
    assert "QuickEmbed" in quick[0]["text"] and again == quick and quick != tiny


def test_key_parts():
    assert normalize("  What is the MAIN result?? ") == "what is the main result"
    assert make_key("a", 1) == make_key("a", 1) != make_key("a", 2) and len(make_key("x")) == 64
    assert search_key("q", "hash-a", 4) != search_key("q", "hash-b", 4) != search_key("q", "hash-b", 8)


def test_semantic_cache_thresholds(day3):
    """The 6.4 calibration, as the notebook computes it: loose hits the pass@10 near-miss, recommended doesn't."""
    cache = SemanticCache(lambda texts: day3.embed_texts(texts), threshold=1.0, scope="tinycoder")
    assert cache.put(Q.PASS1, "41% on the first attempt [p4-c1].", passed_guards=True)
    scores = {q: cache.lookup(q).score for q in (Q.REWORDED, Q.NEAR_MISS)}
    assert scores[Q.REWORDED] > scores[Q.NEAR_MISS]
    loose = round(min(scores.values()) - 0.05, 2)
    recommended = round((scores[Q.REWORDED] + scores[Q.NEAR_MISS]) / 2, 2)
    cache.threshold = loose
    assert cache.lookup(Q.NEAR_MISS).answer == "41% on the first attempt [p4-c1]."      # the failure
    cache.threshold = recommended
    assert cache.lookup(Q.REWORDED).answer and cache.lookup(Q.NEAR_MISS).answer is None
    assert (round(scores[Q.REWORDED], 3), round(scores[Q.NEAR_MISS], 3)) == (0.713, 0.444)   # recorded in the notes


def test_failed_guard_answers_are_never_cached(day3):
    cache = SemanticCache(lambda texts: day3.embed_texts(texts), threshold=0.0, scope="tinycoder")
    failed = AIMessage(CITATION_FAILED.format(ids=["c99"]), response_metadata={"guard_failed": "citation_check"})
    limited = AIMessage("Model call limits exceeded: run limit (6/6)")
    for message in (failed, limited):
        assert not answer_passed_guards(message)
        assert cache.put(Q.PASS1, message.text, passed_guards=answer_passed_guards(message)) is False
    assert cache.lookup(Q.PASS1).answer is None and cache.questions == []
    assert answer_passed_guards(AIMessage("41% [p4-c1]."))


def test_cached_input_tokens_never_raises():
    assert cached_input_tokens(AIMessage("x")) is None
    no_details = AIMessage("x", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})
    assert cached_input_tokens(no_details) is None
    assert cached_input_tokens(SimpleNamespace(usage_metadata={"input_token_details": "garbage"})) is None
    assert cached_input_tokens(None) is None and cached_input_tokens(object()) is None
    with_cache = AIMessage("x", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6,
                                                "input_token_details": {"cache_read": 3}})
    assert cached_input_tokens(with_cache) == 3


def test_provider_prompt_cache_on_a_shared_prefix(day3, mock):
    from paper_agent.agent.papers import PAPERS
    prefix = AGENT_SYSTEM_V1 + "\n\nThe whole paper:\n" + "\n\n".join(c["text"] for c in PAPERS["tinycoder"]["chunks"])
    model = make_chat_model()
    first, second = (model.invoke([{"role": "system", "content": prefix}, {"role": "user", "content": q}])
                     for q in ("What is the main result?", "What are the limitations?"))
    assert cached_input_tokens(first) == 0
    assert 1024 <= cached_input_tokens(second) <= second.usage_metadata["input_tokens"]
    assert cached_input_tokens(second) % 128 == 0
