"""Papers the service has seen (Day 4, 4.4): the id is a hash of the content, so a paper is parsed and embedded
once, however many people upload it and whatever they call it.

Built on Days 2-3: register_paper() (validate, parse, clean, chunk; no API call), Day 2's embedding cache and
Chroma index. What is new is the id, the spans and search(), which search_paper calls (6.3 slows it down).
The PDFs and the index live on disk (data_dir), so a restarted api or a second worker finds them again.
"""
import hashlib
import time
from pathlib import Path
from typing import Optional

from paper_agent.agent.papers import PAPERS, register_paper
from paper_agent.rag import pdf_rag as day2
from paper_agent.service.gateway import GatewayEmbedder
from paper_agent.service.tracing import TRACER

SEARCH_DELAY_S = 0.0          # 6.3: pretend the vector database is overloaded


def setup_retrieval(router=None, data_dir=".", embed_model: Optional[str] = None) -> None:
    """Point Day 2's retrieval at the gateway: embeddings through the "paper-embed" alias, a Chroma index in
    data_dir. router=None: the LiteLLM proxy (proxy mode)."""
    import os
    day2.llm = GatewayEmbedder(router)
    day2.EMBED_MODEL = embed_model or os.environ.get("EMBED_MODEL") or "paper-embed"
    day2.collection = day2.open_index(str(Path(data_dir) / "index"))


def paper_id_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


class PaperStore:
    def __init__(self, data_dir="."):
        self.folder = Path(data_dir) / "papers"
        self.cache_dir = Path(data_dir) / "embedding_cache"
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, paper_id: str) -> Path:
        return self.folder / f"{paper_id}.pdf"

    def ingest(self, data: bytes) -> tuple:
        """(paper_id, is_new). Validate the bytes first (validate_upload): this trusts them."""
        paper_id = paper_id_of(data)
        if self.get(paper_id):
            return paper_id, False
        path = self.path(paper_id)
        path.write_bytes(data)
        with TRACER.start_as_current_span("parse_pdf") as span:
            paper = register_paper(path, paper_id)
            span.set_attribute("pdf.chunks", len(paper["chunks"]))
        with TRACER.start_as_current_span("embed_chunks") as span:
            vectors = day2.embed_chunks(paper["chunks"], path, cache_dir=str(self.cache_dir))
            day2.add_to_index(day2.collection, paper["chunks"], vectors, paper_id)
            span.set_attribute("embed.model", day2.EMBED_MODEL)
        return paper_id, True

    def get(self, paper_id: str) -> Optional[dict]:
        """The paper, if it was ever ingested: from memory, or re-registered from disk (no API call)."""
        if paper_id in PAPERS:
            return PAPERS[paper_id]
        path = self.path(paper_id)
        if path.exists() and day2.index_model(day2.collection, paper_id):
            return register_paper(path, paper_id)
        return None


def search(query: str, paper_id: str, k: int = 4) -> list:
    """Day 2's retriever, in a `retrieve` span that records which chunks came back."""
    with TRACER.start_as_current_span("retrieve") as span:
        if SEARCH_DELAY_S:
            time.sleep(SEARCH_DELAY_S)
        hits = day2.retrieve(query, paper_id=paper_id, k=k)
        span.set_attribute("retrieve.chunk_ids", [h["id"] for h in hits])
        return hits
