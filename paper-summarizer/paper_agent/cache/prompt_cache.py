"""Provider prompt caching (Day 3, 6.5): nothing to store on our side, just put the stable part first.

Providers that support it bill a long prefix they have seen recently (system prompt, tool definitions,
the paper) at a lower rate, and report it as cached input tokens. Support and reporting vary: some do it
automatically above a minimum length, some need the prefix marked, some free endpoints report nothing.
"""
from typing import Optional


def cached_input_tokens(ai_message) -> Optional[int]:
    """Cached input tokens the provider reported for this reply; None if it reported nothing. Never raises."""
    try:
        details = (getattr(ai_message, "usage_metadata", None) or {}).get("input_token_details") or {}
        value = details.get("cache_read")
        return None if value is None else int(value)
    except Exception:
        return None
