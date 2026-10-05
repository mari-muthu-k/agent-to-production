"""The workshop's sample papers: real arXiv papers under CC BY 4.0 (see data/papers/MANIFEST.md).

fetch_paper("mistral-7b") returns a local path, trying in order:
  1. data/papers/ in this repo (Docker, local runs)
  2. the course GitHub repo (Colab: fast, no arXiv rate limits for 200 people at once)
  3. arxiv.org, pinned to the exact version whose licence was checked
and verifies the sha256 so every participant indexes the identical file.
"""
import hashlib
import json
import os
import urllib.request
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = REPO_ROOT / "data" / "papers"


@cache
def registry() -> dict:
    return json.loads(resources.files("paper_agent").joinpath("data/papers.json").read_text())


def paper_info(paper_id: str) -> dict:
    for p in registry()["papers"]:
        if p["id"] == paper_id:
            return p
    raise KeyError(f"unknown paper {paper_id!r}; known: {[p['id'] for p in registry()['papers']]}")


def paper_ids() -> list:
    return [p["id"] for p in registry()["papers"]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_paper(paper_id: str, dest: Optional[Path] = None) -> Path:
    info = paper_info(paper_id)
    local = LOCAL_DIR / info["file"]
    if local.exists() and _sha256(local) == info["sha256"]:
        return local
    dest = Path(dest or os.environ.get("PAPER_AGENT_DATA", "data")) / "papers"
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / info["file"]
    if target.exists() and _sha256(target) == info["sha256"]:
        return target
    urls = [f"{registry()['repo_raw']}/{info['file']}", f"https://arxiv.org/pdf/{info['arxiv']}"]
    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-agent-workshop/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                target.write_bytes(resp.read())
            if _sha256(target) == info["sha256"]:
                return target
            errors.append(f"{url}: checksum mismatch")
        except Exception as e:   # try the next source
            errors.append(f"{url}: {type(e).__name__}: {e}")
    raise RuntimeError(f"could not download {paper_id}: " + "; ".join(errors))


def edge_case_pdf(name: str) -> Path:
    """Path of a generated (FICTIONAL) edge-case PDF shipped with the package: two_column, image_only,
    hidden_white, hidden_tiny."""
    path = Path(str(resources.files("paper_agent").joinpath(f"fixtures/pdfs/{name}.pdf")))
    if not path.exists():
        raise FileNotFoundError(f"no edge-case PDF {name!r}; run tools/make_edge_case_pdfs.py")
    return path
