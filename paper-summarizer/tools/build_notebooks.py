"""Generate the workshop notebooks from templates (Day 2 onwards).

Notebooks live at the course repo root, <repo>/notebooks/dayN/{Practice,Demo}.ipynb, so they can be
opened straight from GitHub in Google Colab. Override the location with NOTEBOOKS_DIR.

Day 1 is edited by hand (in Colab) and is never touched by this script.
Days 2+ are generated from notebook_templates/dayN.py: edit the template, then `make notebooks`.

    python tools/build_notebooks.py 3        # write the Day 3 notebooks
    python tools/build_notebooks.py --check  # fail if a committed notebook differs from its template

Only the days you name are written. Day 2's notebooks have since been edited in Colab, so regenerating
them would overwrite those edits.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "notebook_templates")]
NOTEBOOKS = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks")
GENERATED_DAYS = [2, 3]
# Days whose committed notebooks must equal their template (tests/test_tools.py). Day 2's notebooks were
# edited in Colab after generation, so they are no longer checked.
TEMPLATE_DAYS = [3]


def targets(days=GENERATED_DAYS) -> list:
    """[(path, Notebook builder)] for every generated notebook of `days`."""
    out = []
    for day in days:
        mod = importlib.import_module(f"day{day}")
        out += [(NOTEBOOKS / f"day{day}" / "Practice.ipynb", mod.practice),
                (NOTEBOOKS / f"day{day}" / "Demo.ipynb", mod.demo)]
    return out


def render(builder) -> str:
    import nbformat
    return nbformat.writes(builder().build()) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("days", nargs="*", type=int, help="days to build (default: all generated days)")
    parser.add_argument("--check", action="store_true", help="only compare, do not write")
    args = parser.parse_args()
    if not args.days and not args.check:
        parser.error(f"name the days to write, e.g. `python tools/build_notebooks.py 3` (generated: {GENERATED_DAYS})")
    stale = []
    for path, builder in targets(args.days or TEMPLATE_DAYS):          # --check alone: the template-checked days
        text = render(builder)
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(NOTEBOOKS.parent)}")
    for path in stale:
        print(f"STALE {path}: run `make notebooks`")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
