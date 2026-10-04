"""Generate the workshop notebooks.

Notebooks live at the course repo root, <repo>/notebooks/dayN/{Practice,Demo}.ipynb, so they can be
opened straight from GitHub in Google Colab. Override the location with NOTEBOOKS_DIR.

Day 1   byte-for-byte copies of the reference notebooks (never edited), renamed:
          Day1_CodeAlong.ipynb       -> day1/Practice.ipynb  (participants, with the instructor)
          Day1_Instructor_Demo.ipynb -> day1/Demo.ipynb      (instructor only)
        Checksums are recorded in notebooks/day1/CHECKSUMS.json so tests catch accidental edits.
Day 2-4 generated from package code + templates (added in M1-M3).

    python tools/build_notebooks.py                       # verify Day 1 copies against CHECKSUMS.json
    python tools/build_notebooks.py --reference ../reference   # (re)copy Day 1 from the reference folder
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks")
DAY1 = NOTEBOOKS / "day1"
# reference file name -> published name
DAY1_FILES = {"Day1_CodeAlong.ipynb": "Practice.ipynb", "Day1_Instructor_Demo.ipynb": "Demo.ipynb"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_day1(reference) -> int:
    DAY1.mkdir(parents=True, exist_ok=True)
    checksum_file = DAY1 / "CHECKSUMS.json"
    if reference:
        for src, dst in DAY1_FILES.items():
            shutil.copyfile(Path(reference) / src, DAY1 / dst)
        sums = {dst: sha256(DAY1 / dst) for dst in DAY1_FILES.values()}
        checksum_file.write_text(json.dumps(sums, indent=2) + "\n")
        print(f"day1: copied {len(DAY1_FILES)} notebooks from {reference}")
        return 0
    sums = json.loads(checksum_file.read_text())
    names = list(DAY1_FILES.values())
    bad = [n for n in names if sha256(DAY1 / n) != sums.get(n)]
    for n in names:
        print(f"day1: {n}: {'MODIFIED' if n in bad else 'ok'}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", help="folder holding the Day 1 reference notebooks")
    args = parser.parse_args()
    return build_day1(args.reference)


if __name__ == "__main__":
    sys.exit(main())
