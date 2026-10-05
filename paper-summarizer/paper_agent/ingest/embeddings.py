"""Embedders (Day 2, section 5.3): text -> L2-normalised float32 vectors.

OpenAIEmbedder   EMBED_MODEL through the same OpenAI-compatible endpoint as chat (LLM_BASE_URL).
                 Same reliability as Day 1: retries only on timeouts/429/5xx with backoff + Retry-After,
                 circuit breaker, and one log record per call in llm_client.CALL_LOG.
HashingEmbedder  local, free, offline, no model: hashed bag-of-words (see hashing.py). Useful to
                 teach the mechanism, and to show what happens when index and query use different models.

Every embedder has a `model` name. Stores remember it and refuse queries from a different one.
"""
import os
import time
import uuid
from typing import Optional

import numpy as np

from paper_agent.ingest.hashing import hashing_embed
from paper_agent.llm_client import RETRYABLE_ERRORS, CircuitOpenError, LLMClient, LLMConfig


class EmbeddingMismatchError(Exception):
    """Query vectors come from a different embedding model than the index."""


def normalise(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    def __init__(self, dim: int = 256, salt: str = "local"):
        self.dim, self.salt = dim, salt
        self.model = f"hashing-{salt}-{dim}"
        self.tokens_used = 0

    def embed(self, texts: list) -> np.ndarray:
        return np.stack([hashing_embed(t, self.dim, self.salt) for t in texts]) if texts else np.zeros((0, self.dim))

    __call__ = embed


class OpenAIEmbedder:
    def __init__(self, model: Optional[str] = None, client=None, batch_size: int = 64,
                 dimensions: Optional[int] = None, price_per_1m: float = 0.0, config: Optional[LLMConfig] = None):
        self.model = model or os.environ.get("EMBED_MODEL") or "text-embedding-3-small"
        self.batch_size, self.dimensions = batch_size, dimensions
        # Reuse Day 1's LLMClient for its retry policy, breaker, call log and spend tracking.
        cfg = config or LLMConfig(model=self.model, price_in_per_1m=price_per_1m)
        self.policy = LLMClient(cfg, client=client)
        self.client = self.policy.client
        self.tokens_used = 0
        self._cache: dict = {}

    def _create(self, batch: list):
        params = {"model": self.model, "input": batch}
        if self.dimensions:
            params["dimensions"] = self.dimensions
        attempts, start, request_id = 0, time.perf_counter(), uuid.uuid4().hex[:8]
        while True:
            if not self.policy.breaker.allow():
                raise CircuitOpenError("embedding provider is failing; circuit open, failing fast")
            attempts += 1
            try:
                resp = self.client.embeddings.create(**params)
                self.policy.breaker.record_success()
                break
            except RETRYABLE_ERRORS as exc:
                self.policy.breaker.record_failure()
                if attempts > self.policy.config.max_retries:
                    raise
                time.sleep(max(self.policy._backoff_delay(attempts - 1), self.policy._retry_after(exc)))
        n = getattr(resp.usage, "prompt_tokens", 0) or 0
        self.tokens_used += n
        self.policy._record(request_id=request_id, model=self.model, status="ok", attempts=attempts,
                            input_tokens=n, output_tokens=0, finish_reason=None,
                            latency_ms=int((time.perf_counter() - start) * 1000),
                            cost_usd=round(self.policy._cost(n, 0), 6))
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def embed(self, texts: list) -> np.ndarray:
        """Embed many texts in batches. Identical texts are embedded once per embedder (cache)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            for text, vec in zip(batch, self._create(batch), strict=True):
                self._cache[text] = vec
        if not texts:
            return np.zeros((0, self.dimensions or 0), dtype=np.float32)
        return normalise([self._cache[t] for t in texts])

    __call__ = embed
