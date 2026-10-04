"""Versioned prompts. Files are named <name>_<version>.txt and loaded verbatim (no strip)."""
from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


@cache
def load_prompt(name: str, version: str = "v1") -> str:
    return (PROMPT_DIR / f"{name}_{version}.txt").read_text(encoding="utf-8")


def prompt_id(name: str, version: str = "v1") -> str:
    """Short label to log with each call, e.g. 'explainer_v1'."""
    return f"{name}_{version}"
