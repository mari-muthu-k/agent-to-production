"""Offline test doubles from Day 1, section 6.4: instant, free, no API calls."""
from types import SimpleNamespace as NS

from paper_agent.llm_client import TransientError


def fake_response(content, finish_reason="stop"):
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish_reason)],
              usage=NS(prompt_tokens=100, completion_tokens=20))


class FakeLLM:
    """Pretends to be the OpenAI client; returns or raises scripted items in order. No network."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.chat = NS(completions=NS(create=self._create))
    def _create(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FlakyProvider:
    """Wraps the real client; the first `failures` calls raise a temporary error (like a 429)."""
    def __init__(self, real_client, failures=2, retry_after=None):
        self.real, self.remaining, self.retry_after = real_client, failures, retry_after
        self.chat = NS(completions=NS(create=self._create))
    def _create(self, **kwargs):
        if self.remaining > 0:
            self.remaining -= 1
            raise TransientError("simulated 429 Too Many Requests", retry_after=self.retry_after)
        return self.real.chat.completions.create(**kwargs)
