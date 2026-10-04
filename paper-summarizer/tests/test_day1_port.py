"""Drift guards: the package must match the Day 1 reference notebook it was ported from."""
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

import paper_agent.llm_client as llm_client
from paper_agent import schemas
from paper_agent.explain import EXPLAINER_PROMPT
from paper_agent.fixtures.tinycoder import PAPER

ROOT = Path(__file__).resolve().parents[1]
DAY1 = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks") / "day1"
CHECKSUMS = json.loads((DAY1 / "CHECKSUMS.json").read_text())


def cells(name: str) -> list:
    return ["".join(c["source"]) for c in json.loads((DAY1 / name).read_text())["cells"]]


CODEALONG = cells("Practice.ipynb")


def cell_with(marker: str) -> str:
    return next(c for c in CODEALONG if marker in c)


@pytest.mark.parametrize("name", sorted(CHECKSUMS))
def test_day1_notebooks_are_unmodified_copies(name):
    assert hashlib.sha256((DAY1 / name).read_bytes()).hexdigest() == CHECKSUMS[name]


def test_llm_client_matches_the_solution_cell():
    solution = cell_with("# 🆘 SOLUTION VERSION").split("\n", 2)[2]
    assert Path(llm_client.__file__).read_text().rstrip("\n") == solution.rstrip("\n")


def test_explainer_prompt_matches_the_catch_up_cell():
    src = cell_with("6.0 CATCH-UP")
    assert EXPLAINER_PROMPT == re.search(r'EXPLAINER_PROMPT = """(.*?)"""', src, re.S).group(1)


def test_paper_fixture_matches_the_notebook():
    ns: dict = {}
    exec(cell_with('PAPER = {\n    "title"').split("SECTION_IDS")[0], ns)   # a dict literal, nothing else
    assert ns["PAPER"] == PAPER


def test_schemas_match_the_notebook():
    ns: dict = {}
    exec(cell_with("class PaperExplainer(BaseModel)").replace('print("✅ schemas defined")', ""), ns)
    for name in ("Answer", "KeyTerm", "PaperExplainer"):
        assert ns[name].model_json_schema() == getattr(schemas, name).model_json_schema(), name
