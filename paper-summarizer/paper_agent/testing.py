"""Offline test doubles from Day 1, section 3.4: instant, free, no API calls."""
from paper_agent.llm_client import TransientError


def fake_response(content, finish_reason="stop", **message_fields):
    """A /chat/completions reply as OpenRouter returns it: a plain dict."""
    return {"choices": [{"message": {"role": "assistant", "content": content, **message_fields},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def fake_embedding_response(vectors, usage=True):
    """An /embeddings reply. usage=False: the provider sent no token counts (Gemini does this)."""
    reply = {"data": [{"index": i, "embedding": list(v)} for i, v in enumerate(vectors)]}
    if usage:
        reply["usage"] = {"prompt_tokens": 10 * len(vectors), "total_tokens": 10 * len(vectors)}
    return reply


class FakeLLM:
    """Pretends to be the HTTP client; returns or raises scripted items in order. No network.
    The same script serves chat() and embeddings(); `bodies` keeps every request body."""
    def __init__(self, script):
        self.script, self.calls, self.bodies = list(script), 0, []
    def chat(self, **body):
        self.calls += 1
        self.bodies.append(body)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    embeddings = chat


class FlakyProvider:
    """Wraps the real client; the first `failures` calls raise a temporary error (like a 429)."""
    def __init__(self, real_client, failures=2, retry_after=None):
        self.real, self.remaining, self.retry_after = real_client, failures, retry_after
    def chat(self, **body):
        if self.remaining > 0:
            self.remaining -= 1
            raise TransientError("simulated 429 Too Many Requests", retry_after=self.retry_after)
        return self.real.chat(**body)
