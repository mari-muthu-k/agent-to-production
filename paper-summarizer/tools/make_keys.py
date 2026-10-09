"""`make keys`: demo virtual keys for alice, bob and eval-bot on the local LiteLLM proxy.

Deletes any existing key with the same alias, generates a new one, prints it and saves {alias: key} to
keys.local.json (gitignored). Run again any time: the old keys stop working.

    python tools/make_keys.py            # in the tools container: GATEWAY_URL=http://gateway:4000
    python tools/make_keys.py --delete   # remove the demo keys (after class)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_agent.service.keys import DEMO_KEYS, KEYS_FILE, delete_keys, recreate_keys  # noqa: E402


def main(argv) -> int:
    if "--delete" in argv:
        print(f"deleted {delete_keys(list(DEMO_KEYS))} demo key(s)")
        return 0
    keys = recreate_keys()
    for alias, key in keys.items():
        s = DEMO_KEYS[alias]
        print(f"{alias:9} {key}   budget ${s['max_budget']}/{s['budget_duration']}, {s['rpm_limit']} requests/min")
    print(f"\nsaved to {KEYS_FILE.name} (gitignored). These are local demo keys: the api takes them as "
          "'Authorization: Bearer <key>'.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
