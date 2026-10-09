"""Virtual keys on the LiteLLM proxy (Day 4, 5.6 in production): one key per user, with a budget, a reset
period and a requests-per-minute limit. Used by `make keys`, the instructor demo notebook and the gateway tests.

The proxy stores only a hash of each key, so an existing key can't be read back: to make key creation
repeatable we delete any key with the same alias and generate a new one. Keys are shown masked.
"""
import json
import os
from pathlib import Path
from typing import Optional

import httpx

KEYS_FILE = Path(os.environ.get("KEYS_FILE", Path(__file__).resolve().parents[2] / "keys.local.json"))
MODELS = ["paper-explainer", "paper-explainer-fallback", "paper-embed"]
DEMO_KEYS = {   # alias: settings. bob's budget is tiny on purpose: about one and a half questions on the mock.
    "alice": {"user_id": "alice", "max_budget": 1.00, "budget_duration": "1d", "rpm_limit": 10},
    "bob": {"user_id": "bob", "max_budget": 0.0005, "budget_duration": "1d", "rpm_limit": 10},
    "eval-bot": {"user_id": "eval-bot", "max_budget": 5.00, "budget_duration": "1d", "rpm_limit": 600},
}


def gateway_url() -> str:
    return os.environ.get("GATEWAY_URL", "http://gateway:4000").rstrip("/")


def admin(method: str, path: str, **kwargs) -> httpx.Response:
    """A call to the proxy's admin API with the master key (from the environment, never printed)."""
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"}
    return httpx.request(method, f"{gateway_url()}{path}", headers=headers, timeout=30, **kwargs)


def mask(key: Optional[str]) -> str:
    return f"{key[:5]}…{key[-4:]}" if key and len(key) > 12 else "***"


def delete_keys(aliases: list) -> int:
    """Delete every key with one of these aliases. Returns how many were deleted."""
    r = admin("POST", "/key/delete", json={"key_aliases": list(aliases)})
    if r.status_code == 200:
        return len(r.json().get("deleted_keys") or [])
    if r.status_code == 404:                     # no key with these aliases: nothing to delete
        return 0
    r.raise_for_status()
    return 0


def generate_key(alias: str, **settings) -> dict:
    """POST /key/generate. Returns the proxy's reply (its "key" is the only time the raw key is visible)."""
    body = {"key_alias": alias, "models": MODELS, **settings}
    r = admin("POST", "/key/generate", json=body)
    if r.status_code != 200:
        raise RuntimeError(f"/key/generate {alias}: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def recreate_keys(specs: dict = DEMO_KEYS, path: Path = KEYS_FILE) -> dict:
    """Idempotent: delete the aliases, generate fresh keys, save {alias: key} to keys.local.json (gitignored)."""
    delete_keys(list(specs))
    keys = {alias: generate_key(alias, **settings)["key"] for alias, settings in specs.items()}
    saved = load_keys(path)
    saved.update(keys)
    path.write_text(json.dumps(saved, indent=2) + "\n")
    return keys


def load_keys(path: Path = KEYS_FILE) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def key_info(key: str) -> dict:
    """GET /key/info for one key (the key itself may ask)."""
    r = httpx.get(f"{gateway_url()}/key/info", params={"key": key}, headers={"Authorization": f"Bearer {key}"},
                  timeout=30)
    r.raise_for_status()
    return r.json()["info"]
