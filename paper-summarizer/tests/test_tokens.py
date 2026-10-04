"""Token counting works offline (encodings baked into the image) and budgets are respected."""
from types import SimpleNamespace as NS

from paper_agent.fixtures.tinycoder import PAPER
from paper_agent.tokens import (
    cost_usd,
    count_message_tokens,
    count_tokens,
    estimate_tokens_from_words,
    fit_sections,
    truncate_to_tokens,
)


def test_count_tokens_offline():
    assert count_tokens("Researchers trained a small code model.") == 7
    assert count_tokens("Researchers trained a small code model.", "gpt2") >= 7


def test_message_tokens_include_overhead():
    msgs = [{"role": "user", "content": "hello"}]
    assert count_message_tokens(msgs) == count_tokens("hello") + 4 + 3


def test_fit_sections_respects_budget():
    total = sum(count_tokens(t) for t in PAPER["sections"].values())
    assert fit_sections(PAPER, total) == list(PAPER["sections"])
    chosen = fit_sections(PAPER, 100)
    assert chosen and sum(count_tokens(PAPER["sections"][s]) for s in chosen) <= 100


def test_truncate_and_estimates():
    assert count_tokens(truncate_to_tokens(PAPER["sections"]["abstract"], 10)) <= 10
    assert estimate_tokens_from_words(8000) == 10666
    assert cost_usd(NS(prompt_tokens=1_000_000, completion_tokens=500_000), 0.5, 2.0) == 1.5
