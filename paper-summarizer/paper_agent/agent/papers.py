"""The papers the agent can read, and who it is talking to (Day 3, 3.0).

Tools never read globals for "the current paper" or "the current user": both arrive in every run as
`context=Context(user_id=..., paper_id=...)` (create_agent's context_schema), so one process can serve
several papers and readers at once. PAPERS holds what each paper's tools need; the vectors live in
Day 2's Chroma collection, filtered by paper_id.
"""
import re
from dataclasses import dataclass

from paper_agent.rag import pdf_rag as day2


@dataclass
class Context:
    user_id: str
    paper_id: str = "tinycoder"


PAPERS: dict = {}


def section_id(heading: str) -> str:
    """'6 Limitations' -> 'limitations', '3 Training and evaluation' -> 'training-and-evaluation'."""
    bare = re.sub(r"^\d{1,2}(\.\d{1,2})*\.?\s+", "", heading)
    return re.sub(r"[^a-z0-9]+", "-", bare.lower()).strip("-")


def register_paper(pdf, paper_id: str) -> dict:
    """Yesterday's pipeline (validate, parse, clean, chunk), plus what today's tools need: chunks grouped
    by section, a hash of the file's content, and the lines the parser found hidden. No API calls."""
    day2.validate_upload(pdf)
    chunks = day2.chunk_pages(day2.clean_pages(day2.parse_pages(pdf)))
    sections: dict = {}
    for c in chunks:
        sections.setdefault(section_id(c["section"]), []).append(c)
        for line in c["text"].splitlines():                   # a heading that starts inside a chunk
            if day2.is_heading(line) and section_id(line) not in (section_id(c["section"]), "references"):
                sections.setdefault(section_id(line), []).append(c)
    PAPERS[paper_id] = {"id": paper_id, "pdf": str(pdf), "hash": day2.file_sha256(pdf), "chunks": chunks,
                        "sections": sections, "hidden": day2.hidden_text(pdf)}
    return PAPERS[paper_id]


def load_paper(pdf, paper_id: str) -> dict:
    """register_paper(), then embed the chunks (cached by file hash) and add them to the Chroma index."""
    paper = register_paper(pdf, paper_id)
    day2.add_to_index(day2.collection, paper["chunks"], day2.embed_chunks(paper["chunks"], pdf), paper_id)
    return paper
