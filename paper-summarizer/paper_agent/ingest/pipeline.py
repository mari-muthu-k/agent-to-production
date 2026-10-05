"""The whole ingestion pipeline in one call: PDF -> pages -> blocks -> chunks -> vectors -> store."""
from dataclasses import dataclass, field
from typing import Literal, Optional

from paper_agent.ingest.cleaning import to_blocks
from paper_agent.ingest.loaders import load_pdf
from paper_agent.ingest.splitters import chunk_stats, fixed_chunks, semantic_chunks, sentence_chunks

Strategy = Literal["sentence", "fixed", "semantic"]


@dataclass
class IngestReport:
    paper_id: str
    title: str
    pages: int
    sections: list
    chunks: int
    stats: dict
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.paper_id}: {self.title[:70]}",
                 f"  {self.pages} pages, {len(self.sections)} sections, {self.chunks} chunks "
                 f"(median {self.stats['median_chars']} chars)",
                 f"  sections: {', '.join(self.sections[:12])}{' ...' if len(self.sections) > 12 else ''}"]
        lines += [f"  ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


def make_chunks(blocks: list, paper_id: str, strategy: Strategy = "sentence", embedder=None, **kwargs) -> list:
    if strategy == "fixed":
        return fixed_chunks(blocks, paper_id, **kwargs)
    if strategy == "semantic":
        return semantic_chunks(blocks, paper_id, embedder, **kwargs)
    return sentence_chunks(blocks, paper_id, **kwargs)


def ingest_pdf(path, embedder, store, paper_id: Optional[str] = None, title: Optional[str] = None,
               strategy: Strategy = "sentence", layout: str = "columns", **chunk_kwargs):
    """Load, chunk, embed and store one PDF. Returns (report, chunks)."""
    if paper_id and not title:
        try:
            from paper_agent.papers import paper_info
            title = paper_info(paper_id)["title"]
        except KeyError:
            pass
    doc = load_pdf(path, paper_id=paper_id, title=title, layout=layout)
    blocks = to_blocks(doc.pages)
    chunks = make_chunks(blocks, doc.paper_id, strategy, embedder, **chunk_kwargs)
    if chunks:
        store.add(chunks, embedder.embed([c.text for c in chunks]), embed_model=embedder.model)
    sections = list(dict.fromkeys(b.section for b in blocks))
    warnings = list(doc.warnings)
    if not chunks:
        warnings.append("no text extracted at all: is this a scanned PDF?")
    report = IngestReport(doc.paper_id, doc.title, len(doc.pages), sections, len(chunks), chunk_stats(chunks),
                          warnings)
    return report, chunks
