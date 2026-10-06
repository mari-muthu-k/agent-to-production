"""Copy paper_agent/llm_client.py into the two `%%writefile llm_client.py` cells of Day 1 Practice.

The package file is the SOLUTION cell; the TODO cell is derived from it by blanking out TODO 1
(backoff) and TODO 2 (truncation). Everything else in the notebook is left untouched.

    python tools/sync_day1_llm_client.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks") / "day1" / "Practice.ipynb"

TODO_1 = ('''        exp = min(self.config.max_delay_s, self.config.base_delay_s * (2 ** attempt))
        return random.uniform(0, exp)''', '''        # ✏️ TODO 1: replace the fixed delay with exponential backoff + full jitter.
        #   exp = base_delay_s * (2 ** attempt), capped at max_delay_s
        #   (use self.config.base_delay_s and self.config.max_delay_s)
        #   then return random.uniform(0, exp)
        return 1.0''')
TODO_2 = ('''        if truncated:
            raise TruncatedOutputError(f"[{request_id}] output cut off at max_tokens={max_tokens}{hint}")
''', '''        # ✏️ TODO 2: if `truncated` is True, raise TruncatedOutputError
        #   instead of returning half an answer. Two lines.

''')


def main() -> int:
    if "--force" not in sys.argv:
        print("Day 1 is frozen: the package llm_client.py now has Day 2 additions (embed(), ...).\n"
              "Syncing would change the Day 1 notebook. Re-run with --force only if that is really intended.")
        return 1
    solution = (ROOT / "paper_agent" / "llm_client.py").read_text()
    todo = solution
    for old, new in (TODO_1, TODO_2):
        if old not in todo:
            print(f"cannot derive the TODO version: {old.strip()[:60]!r} not found")
            return 1
        todo = todo.replace(old, new)
    raw = NOTEBOOK.read_text()
    nb = json.loads(raw)
    cells = [c for c in nb["cells"] if "".join(c["source"]).startswith("%%writefile llm_client.py")]
    if len(cells) != 2:
        print(f"expected 2 %%writefile llm_client.py cells, found {len(cells)}")
        return 1
    cells[0]["source"] = ("%%writefile llm_client.py\n" + todo.rstrip("\n")).splitlines(keepends=True)
    cells[1]["source"] = ("%%writefile llm_client.py\n# SOLUTION VERSION\n"
                          + solution.rstrip("\n")).splitlines(keepends=True)
    out = json.dumps(nb, indent=2, ensure_ascii=False) + ("\n" if raw.endswith("\n") else "")
    NOTEBOOK.write_text(out)
    print(f"synced llm_client.py into {NOTEBOOK.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
