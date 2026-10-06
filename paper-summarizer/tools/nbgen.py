"""A tiny notebook builder for the workshop's generated notebooks (Day 2 onwards).

Conventions it enforces (they match Day 1):
- cells numbered by section ("5.3") to match slide tags;
- every section of a Practice notebook starts with a "⏩ N.0 CATCH-UP" cell that recreates everything
  earlier sections defined. It is assembled from the SAME strings as the teaching cells marked
  `carry=True` (solutions substituted for TODOs), so the two can never drift;
- at most 4 TODOs, each followed by an offline test cell (tag `todo-test`) and a rescue cell whose
  code is the solution string;
- TODO cells carry their solution in `metadata.solution` for tools/exec_notebooks.py.

`src(obj)` returns the source of a package function, so teaching cells show exactly the code that
the package (and its tests) run.
"""
import ast
import inspect
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

COLAB_METADATA = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
MAX_TODOS = 4


def src(*objects) -> str:
    return "\n\n".join(textwrap.dedent(inspect.getsource(o)).strip() for o in objects)


def consts(module, *names) -> str:
    """Source of a module's top-level assignments to `names` (constants, regexes), in module order."""
    source = inspect.getsource(module)
    picked = [node for node in ast.parse(source).body
              if isinstance(node, (ast.Assign, ast.AnnAssign))
              and any(getattr(t, "id", None) in names for t in (node.targets if isinstance(node, ast.Assign)
                                                               else [node.target]))]
    missing = set(names) - {getattr(t, "id", None) for n in picked
                            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])}
    if missing:
        raise KeyError(f"not assigned at top level of {module.__name__}: {sorted(missing)}")
    return "\n".join(ast.get_source_segment(source, n) for n in picked)


def clean(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


class Notebook:
    def __init__(self, name: str, setup: str = ""):
        self.name = name
        self.setup = clean(setup)
        self.cells: list = []
        self.carried: list = []         # source strings recreated by later catch-up cells
        self.todos = 0

    # -- plain cells -----------------------------------------------------------------
    def md(self, text: str):
        self.cells.append(new_markdown_cell(clean(text)))
        return self

    def code(self, source: str, carry: bool = False, tags=()):
        source = clean(source)
        cell = new_code_cell(source)
        if tags:
            cell.metadata["tags"] = list(tags)
        self.cells.append(cell)
        if carry:
            self.carried.append(source)
        return self

    # -- sections and catch-ups -------------------------------------------------------
    def section(self, number: int, title: str, intro: str = "", catch_up: bool = True, preamble: str = ""):
        self.md(f"---\n# Section {number}: {title}" + (f"\n\n{clean(intro)}" if intro else ""))
        if catch_up:
            body = "\n\n".join([p for p in (clean(preamble), self.setup) if p] + self.carried)
            header = (f"# ⏩ {number}.0 CATCH-UP: run ONLY if you joined late or your runtime restarted.\n"
                      "# It recreates everything earlier sections defined, so you can follow along from here.")
            self.code(f"{header}\n{body}\nprint('✅ caught up: ready for Section {number}')", tags=("catch-up",))
        return self

    # -- TODOs ----------------------------------------------------------------------------
    def todo(self, stub: str, solution: str, carry: bool = True):
        self.todos += 1
        if self.todos > MAX_TODOS:
            raise ValueError(f"{self.name}: more than {MAX_TODOS} TODOs")
        cell = new_code_cell(clean(stub))
        cell.metadata["tags"] = ["todo"]
        cell.metadata["solution"] = clean(solution)
        self.cells.append(cell)
        if carry:
            self.carried.append(clean(solution))
        return self

    def todo_test(self, source: str):
        return self.code(source, tags=("todo-test",))

    def rescue(self, solution: str, hint: str):
        self.md(f"<details><summary>🆘 Stuck? Hint</summary>\n\n{clean(hint)}\n</details>\n\n"
                "🆘 **Rescue cell:** tests failing, or you fell behind? Run the next cell, then carry on.")
        return self.code("# 🆘 RESCUE: the finished version. Running it is always safe.\n" + clean(solution),
                         tags=("rescue",))

    # -- output -----------------------------------------------------------------------------
    def build(self):
        nb = new_notebook(cells=self.cells, metadata=dict(COLAB_METADATA))
        for i, cell in enumerate(nb.cells):
            cell.id = f"{self.name}-{i:03d}"          # stable ids: regenerating does not churn the diff
        nbformat.validate(nb)
        return nb

    def write(self, path) -> None:
        nbformat.write(self.build(), str(path))
