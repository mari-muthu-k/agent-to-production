"""Execute every notebook against mock-llm and fail on any unexpected error.

For each Practice (code-along) notebook, three kinds of run, each in a fresh kernel and a scratch folder:
  unsolved   the notebook as participants get it: errors are allowed ONLY in TODO-test cells
  solved     TODO cells replaced by their solutions: zero errors, every TODO test prints ✅
  catch-up   one run per ⏩ CATCH-UP cell: that cell, then everything after it (solved): zero errors
Demo (instructor) notebooks get one plain run: zero errors.

Executed copies are written to build/executed/ for inspection. The solved Day 4 code-along is also saved as
notebooks/day4/executed/Day4_CodeAlong_solved.ipynb: the guide's "executed backup" (gitignored: mock outputs).

    python tools/exec_notebooks.py                 # all notebooks
    python tools/exec_notebooks.py day1            # only notebooks under notebooks/day1
"""
import copy
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks")   # <repo>/notebooks
OUT = ROOT / "build" / "executed"
TIMEOUT_S = int(os.environ.get("NB_CELL_TIMEOUT", "180"))
BACKUPS = {"day4/Day4_CodeAlong.ipynb": "day4/executed/Day4_CodeAlong_solved.ipynb"}


@dataclass
class Spec:
    """How to run one notebook. Day 1 notebooks are never edited, so cells are found by markers."""
    todo_tests: list = field(default_factory=list)        # markers of cells allowed to fail when unsolved
    solutions: list = field(default_factory=list)         # (marker, fn(source, nb) -> solved source)
    solved_must_print: list = field(default_factory=list)


def _day1_solution_cell(src: str, nb) -> str:
    marker = r"%%writefile llm_client.py\n# (🆘 )?SOLUTION VERSION"
    return [c.source for c in nb.cells if re.match(marker, c.source)][-1]


def _day1_todo3(src: str, nb) -> str:
    return re.sub(r"unknown = \[\]\s+# (✏️ )?TODO 3.*", "unknown = set(result.citations) - set(allowed_ids)", src)


SPECS = {
    "day1/Practice.ipynb": Spec(
        todo_tests=["✅ TODO 1: backoff grows"],
        solutions=[("%%writefile llm_client.py\n\"\"\"", _day1_solution_cell), ("unknown = []", _day1_todo3)],
        # TODO 3's cell prints "TODO 3: <error>" when solved and "TODO 3: appendix_b was not caught" when not
        solved_must_print=["✅ TODO 1", "✅ TODO 2", "TODO 3: cites sections that were not provided"],
    ),
    "day1/Demo.ipynb": Spec(),
}


def spec_for(rel: str) -> Spec:
    """Generated notebooks (Day 2+) carry cell tags instead of markers: todo-test, catch-up, solution."""
    return SPECS.get(rel) or Spec()


def is_code(cell) -> bool:
    return cell.cell_type == "code"


def tags(cell) -> set:
    return set(cell.get("metadata", {}).get("tags", []))


def solve(nb, spec: Spec):
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if not is_code(cell):
            continue
        for marker, fn in spec.solutions:
            if marker in cell.source:
                cell.source = fn(cell.source, nb)
        if "solution" in cell.get("metadata", {}):          # generated notebooks: metadata.solution
            cell.source = cell.metadata["solution"]
    return nb


def is_todo_test(cell, spec: Spec) -> bool:
    return "todo-test" in tags(cell) or any(m in cell.source for m in spec.todo_tests)


def is_catch_up(cell, spec: Spec) -> bool:
    first = cell.source.lstrip().splitlines()[0] if cell.source.strip() else ""
    return "catch-up" in tags(cell) or (is_code(cell) and first.startswith("#") and "CATCH-UP" in first)


def execute(nb, label: str, rel: str):
    """Run in a fresh kernel and scratch folder. Returns (executed nb, [(index, cell, error output)])."""
    workdir = ROOT / "build" / "nbexec" / label.replace(" ", "_")
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    client = NotebookClient(nb, timeout=TIMEOUT_S, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": str(workdir)}})
    client.execute()
    out = OUT / label.replace(" ", "_") / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    errors = [(i, c, o) for i, c in enumerate(nb.cells) if is_code(c)
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    return nb, errors


def printed(nb) -> str:
    chunks = []
    for c in nb.cells:
        for o in c.get("outputs", []) if is_code(c) else []:
            chunks.append(o.get("text", "") if o.get("output_type") == "stream"
                          else str(o.get("data", {}).get("text/plain", "")))
    return "\n".join(chunks)


def describe(errors) -> str:
    return "; ".join(f"cell {i} [{c.source.splitlines()[0][:50]!r}] {o['ename']}: {o['evalue'][:80]}"
                     for i, c, o in errors)


def run_notebook(rel: str) -> list:
    """Returns a list of (run label, ok, seconds, detail)."""
    spec = spec_for(rel)
    nb = nbformat.read(NOTEBOOKS / rel, as_version=4)
    has_todos = bool(spec.todo_tests or spec.solutions) or any("todo-test" in tags(c) for c in nb.cells)
    results = []

    def record(label, fn):
        t = time.perf_counter()
        ok, detail = fn()
        results.append((label, ok, time.perf_counter() - t, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<34} {time.perf_counter() - t:6.1f}s  {detail}", flush=True)

    if not has_todos:
        def plain():
            _, errors = execute(copy.deepcopy(nb), f"{rel.replace('/', '-')[:-6]}-full", rel)
            return not errors, describe(errors) or "0 errors"
        record("full run", plain)
        return results

    def unsolved():
        _, errors = execute(copy.deepcopy(nb), f"{rel.replace('/', '-')[:-6]}-unsolved", rel)
        unexpected = [e for e in errors if not is_todo_test(e[1], spec)]
        todo_fails = [e for e in errors if is_todo_test(e[1], spec)]
        if unexpected:
            return False, "unexpected: " + describe(unexpected)
        return True, f"{len(todo_fails)} error(s), all in TODO tests ({describe(todo_fails) or 'none'})"
    record("unsolved (TODOs as given)", unsolved)

    solved_nb = solve(nb, spec)

    must_print = spec.solved_must_print or [f"✅ TODO {i}" for i in
                                            range(1, sum("todo" in tags(c) for c in nb.cells) + 1)]

    def solved():
        executed, errors = execute(copy.deepcopy(solved_nb), f"{rel.replace('/', '-')[:-6]}-solved", rel)
        if rel in BACKUPS and not errors:
            (NOTEBOOKS / BACKUPS[rel]).parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(executed, NOTEBOOKS / BACKUPS[rel])
        text = printed(executed)
        missing = [m for m in must_print if m not in text]
        if errors or missing or "❌ TODO" in text or "was not caught" in text:
            return False, (describe(errors) + f" missing={missing}").strip()
        return True, "0 errors, " + ", ".join(must_print)
    record("solved", solved)

    for i, cell in enumerate(solved_nb.cells):
        if not is_catch_up(cell, spec):
            continue
        name = re.search(r"(\d+\.0) CATCH-UP", cell.source)
        label = f"catch-up {name.group(1) if name else f'cell {i}'} -> end"

        def from_catch_up(i=i, label=label):
            sub = copy.deepcopy(solved_nb)
            sub.cells = [c for j, c in enumerate(sub.cells) if j >= i]
            _, errors = execute(sub, f"{rel.replace('/', '-')[:-6]}-{label}", rel)
            return not errors, describe(errors) or "0 errors (fresh kernel)"
        record(label, from_catch_up)
    return results


def main(argv) -> int:
    only = argv[1] if len(argv) > 1 else ""
    paths = sorted(p.relative_to(NOTEBOOKS).as_posix() for p in NOTEBOOKS.rglob("*.ipynb")
                   if ".ipynb_checkpoints" not in p.parts and "executed" not in p.parts)
    paths = [p for p in paths if p.startswith(only)]
    print(f"LLM_BASE_URL={os.environ.get('LLM_BASE_URL')}  LLM_MODEL={os.environ.get('LLM_MODEL')}")
    if "mock-llm" not in (os.environ.get("LLM_BASE_URL") or ""):
        print("refusing to run: notebooks are executed against mock-llm only (no real tokens spent)")
        return 2
    failures = 0
    for rel in paths:
        print(f"\n{rel}")
        failures += sum(not ok for _, ok, _, _ in run_notebook(rel))
    print(f"\n{'OK' if not failures else 'FAILED'}: {len(paths)} notebook(s), {failures} failing run(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
