"""Generate the workshop notebooks from templates (Day 2 onwards).

Notebooks live at the course repo root, <repo>/notebooks/dayN/{Practice,Demo}.ipynb, so they can be
opened straight from GitHub in Google Colab. Override the location with NOTEBOOKS_DIR.

Day 1 is edited by hand (in Colab) and is never touched by this script.
Days 2+ are generated from notebook_templates/dayN.py: edit the template, then `make notebooks`.

    python tools/build_notebooks.py          # write every generated notebook
    python tools/build_notebooks.py --check  # fail if a committed notebook differs from its template
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "notebook_templates")]
NOTEBOOKS = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks")
GENERATED_DAYS = [2]


def targets() -> list:
    """[(path, Notebook builder)] for every generated notebook."""
    out = []
    for day in GENERATED_DAYS:
        mod = importlib.import_module(f"day{day}")
        out += [(NOTEBOOKS / f"day{day}" / "Practice.ipynb", mod.practice),
                (NOTEBOOKS / f"day{day}" / "Demo.ipynb", mod.demo)]
    return out


def render(builder) -> str:
    import nbformat
    return nbformat.writes(builder().build()) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only compare, do not write")
    args = parser.parse_args()
    stale = []
    for path, builder in targets():
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
