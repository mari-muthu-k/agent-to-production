"""Drift guards: the package must match the code in the Day 1 Practice notebook.

The Day 1 notebooks are edited directly (in Colab), so they are not compared byte for byte. If one
of these tests fails after a notebook edit, decide which side is right and update the other.
"""
import json
import os
import re
from pathlib import Path

import paper_agent.llm_client as llm_client
from paper_agent import schemas
from paper_agent.explain import EXPLAINER_PROMPT
from paper_agent.fixtures.tinycoder import PAPER

ROOT = Path(__file__).resolve().parents[1]
DAY1 = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks") / "day1"


def cells(name: str) -> list:
    return ["".join(c["source"]) for c in json.loads((DAY1 / name).read_text())["cells"]]


CODEALONG = cells("Practice.ipynb")


def cell_with(marker: str) -> str:
    return next(c for c in CODEALONG if marker in c)


def last_cell_matching(pattern: str) -> str:
    return [c for c in CODEALONG if re.search(pattern, c)][-1]


def test_llm_client_is_a_superset_of_the_day1_solution_cell():
    """Day 1's notebook writes its own llm_client.py from this cell, and is never edited again.
    Day 2 extends the package file (embed(), Gemini reasoning, daily-cap errors), so the package must
    keep every class, function and method the Day 1 cell defines, with compatible signatures."""
    import ast
    import inspect
    solution = last_cell_matching(r"^%%writefile llm_client.py\n# (🆘 )?SOLUTION VERSION").split("\n", 2)[2]
    for node in ast.parse(solution).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert hasattr(llm_client, node.name), f"Day 1 defines {node.name}; the package dropped it"
        if isinstance(node, ast.ClassDef):
            cls = getattr(llm_client, node.name)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    assert hasattr(cls, item.name), f"Day 1 defines {node.name}.{item.name}"
                    if isinstance(inspect.getattr_static(cls, item.name), property):
                        continue
                    day1_params = [a.arg for a in item.args.args]
                    now = list(inspect.signature(getattr(cls, item.name)).parameters)
                    assert now[:len(day1_params)] == day1_params, f"{node.name}.{item.name} signature changed"


def test_explainer_prompt_matches_the_catch_up_cell():
    src = last_cell_matching(r"CATCH-UP[\s\S]*EXPLAINER_PROMPT = ")       # the last catch-up has the solved prompt
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
