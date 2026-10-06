"""Precompute every embedding the Day 2 code-along needs, for USE_PRECOMPUTED = True (embedding API down).

Embeds, with the EMBED_MODEL in the environment: the default chunks of tinycoder.pdf, the 8.2 chunks
(120 characters, no overlap), the 8.3 chunks (no cleaning) and every fixed question the notebook asks.
With OTHER_EMBED_MODEL set, the 8.4 question is embedded with that model too. Writes
notebooks/day2/tinycoder_embeddings.<model>.json, recording the model and the vector length.

    EMBED_MODEL=mock-embed python tools/make_precomputed_embeddings.py     # in Docker, against mock-llm
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from paper_agent.llm_client import LLMClient, LLMConfig  # noqa: E402
from paper_agent.rag import pdf_rag  # noqa: E402

DAY2 = Path(os.environ.get("NOTEBOOKS_DIR") or ROOT.parent / "notebooks") / "day2"


def texts_to_embed(pdf: Path) -> list:
    raw = pdf_rag.parse_pages(pdf)
    pages = pdf_rag.clean_pages(raw)
    small = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0, separators=pdf_rag.SEPARATORS)
    texts = [c["text"] for c in pdf_rag.chunk_pages(pages)]
    texts += [c["text"] for c in pdf_rag.chunk_pages(pages, small)]
    texts += [c["text"] for c in pdf_rag.chunk_pages(raw)]
    return list(dict.fromkeys(texts + pdf_rag.QUESTIONS))


def encode(model: str, texts: list, llm: LLMClient) -> dict:
    vectors = np.array(llm.embed(texts, model=model), dtype="<f4")
    return {"model": model, "dim": int(vectors.shape[1]), "texts": texts,
            "vectors": base64.b64encode(vectors.tobytes()).decode()}


def main() -> int:
    model, other = os.environ.get("EMBED_MODEL"), os.environ.get("OTHER_EMBED_MODEL")
    if not model:
        print("set EMBED_MODEL")
        return 1
    pdf = DAY2 / "tinycoder.pdf"
    llm = LLMClient(LLMConfig(max_retries=5, max_delay_s=30))
    sets = [encode(model, texts_to_embed(pdf), llm)]
    if other:
        sets.append(encode(other, [pdf_rag.LONG_FILES], llm))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    out = DAY2 / f"tinycoder_embeddings.{slug}.json"
    out.write_text(json.dumps({"model": model, "dim": sets[0]["dim"], "pdf_sha256": pdf_rag.file_sha256(pdf),
                               "sets": sets}) + "\n")
    print(f"wrote {out.name}: {len(sets[0]['texts'])} texts, vector length {sets[0]['dim']}"
          + (f", plus the 8.4 question with {other}" if other else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
