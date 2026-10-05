"""Vector stores (Day 2, sections 5.4 and 4).

NumpyStore   the whole mechanism in a few lines: a matrix of unit vectors, a dot product, a sort.
             Saved as .npz + .json. Fine up to ~100k chunks.
ChromaStore  the same interface on Chroma, persisted to data/vectorstore. We pass our own vectors
             (no Chroma embedding function), so both stores use exactly the same embeddings.

Both remember which embedding model filled them and raise EmbeddingMismatchError if a query
comes from another model: vectors from different models are not comparable.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from paper_agent.ingest.embeddings import EmbeddingMismatchError
from paper_agent.ingest.splitters import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float     # cosine similarity: 1 = same direction, 0 = unrelated


def top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list:
    """Indices and scores of the k rows most similar to `query` (all rows L2-normalised)."""
    scores = matrix @ query
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


class NumpyStore:
    def __init__(self, embed_model: str):
        self.embed_model = embed_model
        self.chunks: list = []
        self.matrix: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.chunks)

    def _check(self, embed_model: Optional[str]):
        if embed_model and embed_model != self.embed_model:
            raise EmbeddingMismatchError(f"index built with '{self.embed_model}', query embedded with "
                                         f"'{embed_model}': vectors from different models are not comparable")

    def add(self, chunks: list, vectors: np.ndarray, embed_model: Optional[str] = None) -> None:
        self._check(embed_model)
        vectors = np.asarray(vectors, dtype=np.float32)
        known = {c.id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in known]          # adding twice is a no-op
        if not keep:
            return
        self.chunks += [chunks[i] for i in keep]
        new = vectors[keep]
        self.matrix = new if self.matrix is None else np.vstack([self.matrix, new])

    def search(self, query: np.ndarray, k: int = 5, paper_id: Optional[str] = None,
               embed_model: Optional[str] = None) -> list:
        self._check(embed_model)
        if self.matrix is None:
            return []
        if self.matrix.shape[1] != len(query):
            raise EmbeddingMismatchError(f"index vectors have {self.matrix.shape[1]} dimensions, "
                                         f"the query has {len(query)}")
        rows = range(len(self.chunks)) if paper_id is None else \
            [i for i, c in enumerate(self.chunks) if c.paper_id == paper_id]
        rows = list(rows)
        if not rows:
            return []
        hits = top_k(self.matrix[rows], np.asarray(query, dtype=np.float32), k)
        return [Hit(self.chunks[rows[i]], score) for i, score in hits]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), matrix=self.matrix if self.matrix is not None else np.zeros(0))
        path.with_suffix(".json").write_text(json.dumps(
            {"embed_model": self.embed_model, "chunks": [asdict(c) for c in self.chunks]}))

    @classmethod
    def load(cls, path) -> "NumpyStore":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        store = cls(meta["embed_model"])
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.matrix = np.load(path.with_suffix(".npz"))["matrix"] if store.chunks else None
        return store


def collection_name(embed_model: str, prefix: str = "papers") -> str:
    """One collection per embedding model, so vectors from different models never mix."""
    safe = "".join(ch if ch.isalnum() else "-" for ch in embed_model.lower()).strip("-")
    return f"{prefix}-{safe}"[:60]


class ChromaStore:
    def __init__(self, embed_model: str, path="data/vectorstore", collection: Optional[str] = None):
        import chromadb
        from chromadb.config import Settings

        self.embed_model = embed_model
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            collection or collection_name(embed_model), embedding_function=None,
            metadata={"hnsw:space": "cosine", "embed_model": embed_model})
        stored = (self.collection.metadata or {}).get("embed_model")
        if stored and stored != embed_model:
            raise EmbeddingMismatchError(f"collection holds '{stored}' vectors, not '{embed_model}'")

    def __len__(self) -> int:
        return self.collection.count()

    def add(self, chunks: list, vectors: np.ndarray, embed_model: Optional[str] = None) -> None:
        if embed_model and embed_model != self.embed_model:
            raise EmbeddingMismatchError(f"collection is for '{self.embed_model}', got '{embed_model}'")
        if not chunks:
            return
        self.collection.upsert(            # upsert: re-ingesting the same paper does not duplicate it
            ids=[c.id for c in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"paper_id": c.paper_id, "section": c.section, "page": c.page,
                        "pages": ",".join(map(str, c.pages))} for c in chunks])

    def search(self, query: np.ndarray, k: int = 5, paper_id: Optional[str] = None,
               embed_model: Optional[str] = None) -> list:
        if embed_model and embed_model != self.embed_model:
            raise EmbeddingMismatchError(f"collection is for '{self.embed_model}', query used '{embed_model}'")
        n = len(self)
        if n == 0:
            return []
        res = self.collection.query(query_embeddings=[np.asarray(query, dtype=np.float32).tolist()],
                                    n_results=min(k, n), where={"paper_id": paper_id} if paper_id else None,
                                    include=["documents", "metadatas", "distances"])
        hits = []
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                         res["distances"][0], strict=True):
            pages = [int(p) for p in str(meta.get("pages", meta["page"])).split(",") if p]
            chunk = Chunk(cid, meta["paper_id"], text, meta["section"], int(meta["page"]), pages)
            hits.append(Hit(chunk, 1.0 - float(dist)))        # cosine distance -> similarity
        return hits

    def reset(self) -> None:
        name, meta = self.collection.name, self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name, embedding_function=None, metadata=meta)
