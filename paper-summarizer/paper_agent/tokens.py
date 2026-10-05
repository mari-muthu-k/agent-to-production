"""Token counting and context budgeting (Day 1, section 3 demos and 4.3/4.8).

tiktoken's encodings are baked into the Docker image (TIKTOKEN_CACHE_DIR), so this works offline.
Counts are exact for OpenAI-family tokenizers and an estimate for every other provider.
"""
from functools import cache
from typing import Optional

DEFAULT_ENCODING = "o200k_base"     # GPT-4o-era tokenizer
TOKENS_PER_MESSAGE = 4              # role + delimiters overhead per chat message (approximate)
WORDS_TO_TOKENS = 1 / 0.75          # rule of thumb for English (D3.5)


@cache
def _encoding(name: str):
    import tiktoken
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    return len(_encoding(encoding).encode(text or ""))


def count_message_tokens(messages: list, encoding: str = DEFAULT_ENCODING) -> int:
    """Approximate prompt tokens for a chat request: content plus per-message overhead."""
    total = 3   # every reply is primed with an assistant header
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):   # multi-part content: count the text parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        total += TOKENS_PER_MESSAGE + count_tokens(content, encoding)
    return total


def estimate_tokens_from_words(words: int) -> int:
    return int(words * WORDS_TO_TOKENS)


def cost_usd(usage, price_in_per_1m: float, price_out_per_1m: float) -> float:
    """Cost of one call from its `usage` (the dict in an OpenRouter reply; same formula as Day 1)."""
    get = usage.get if isinstance(usage, dict) else (lambda k: getattr(usage, k))
    return (get("prompt_tokens") * price_in_per_1m + get("completion_tokens") * price_out_per_1m) / 1_000_000


def fit_sections(paper: dict, budget_tokens: int, section_ids: Optional[list] = None,
                 encoding: str = DEFAULT_ENCODING) -> list:
    """Section ids, in order, whose text fits within `budget_tokens`. Sections that don't fit are skipped."""
    chosen, used = [], 0
    for sid in section_ids or list(paper["sections"]):
        n = count_tokens(paper["sections"][sid], encoding)
        if used + n <= budget_tokens:
            chosen.append(sid)
            used += n
    return chosen


def truncate_to_tokens(text: str, max_tokens: int, encoding: str = DEFAULT_ENCODING) -> str:
    ids = _encoding(encoding).encode(text or "")
    return text if len(ids) <= max_tokens else _encoding(encoding).decode(ids[:max_tokens])
