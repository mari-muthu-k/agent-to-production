"""The naive RAG baseline used to trigger the five failure modes on purpose (Day 2, section 3).

Fixed-size chunks, no page or section tracking, top-k with no threshold, everything stuffed into
the prompt, and a prompt that never says "you may say you don't know". Do not ship this.
"""
from typing import Optional

import numpy as np

from paper_agent.llm_client import env_models

NAIVE_PROMPT = "Answer the question using the context."


def naive_chunks(text: str, size: int = 300) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)]


def naive_index(chunks: list, embed) -> np.ndarray:
    return embed(chunks)


def naive_search(question: str, chunks: list, matrix: np.ndarray, embed, k: int = 3) -> list:
    q = embed([question])[0]
    scores = matrix @ q
    order = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in order]


def naive_answer(question: str, context_chunks: list, client, model: Optional[str] = None,
                 max_tokens: int = 200) -> str:
    """One raw call to /chat/completions. model=None uses LLM_MODEL (first entry) as set right now."""
    model = model or env_models("LLM_MODEL")[0]
    context = "\n\n".join(f"<chunk>{c}</chunk>" for c in context_chunks)
    reply = client.chat(model=model, max_tokens=max_tokens, temperature=0, messages=[
        {"role": "system", "content": NAIVE_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    return reply["choices"][0]["message"]["content"]
