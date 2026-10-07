"""Precompute every embedding the Day 2 code-along needs, for USE_PRECOMPUTED = True (embedding API down).

Embeds, with the EMBED_MODEL in the environment: the default chunks of tinycoder.pdf, the 8.2 chunks
(120 characters, no overlap), the 8.3 chunks (no cleaning) and every fixed question the notebook asks.
With OTHER_EMBED_MODEL set, the 8.4 question is embedded with that model too. Writes
notebooks/day2/tinycoder_embeddings.<model>.json, recording the model and the vector length.

    EMBED_MODEL=mock-embed python tools/make_precomputed_embeddings.py     # in Docker, against mock-llm

Day 3 (`--day 3`): the chunks of the three Day 3 PDFs, the code-along's fixed questions and the queries
mock-llm's agent sends. Writes notebooks/day3/tinycoder_embeddings_day3.<model>.json. With a real model the
agent writes its own search queries, so USE_PRECOMPUTED can rebuild the index but agent searches need the API.
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
DAY3 = DAY2.parent / "day3"
PDFS = ROOT / "paper_agent" / "fixtures" / "pdfs"


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


def day3_texts() -> list:
    from mock_llm.agent import RUNAWAY_QUERIES
    from paper_agent.agent import questions
    texts = [c["text"] for name in ("tinycoder_day3", "tinycoder_injected", "second_paper")
             for c in pdf_rag.chunk_pages(pdf_rag.clean_pages(pdf_rag.parse_pages(PDFS / f"{name}.pdf")))]
    texts += [v for k, v in vars(questions).items() if k.isupper()]
    texts += ["TinyCoder main result pass@1 compared with the 7B baseline", "files longer than 1,024 tokens",
              "TinyCoder main result", "authors contact email", "limitations", "discussion", *RUNAWAY_QUERIES]
    return list(dict.fromkeys(texts))


def main() -> int:
    model, other = os.environ.get("EMBED_MODEL"), os.environ.get("OTHER_EMBED_MODEL")
    if not model:
        print("set EMBED_MODEL")
        return 1
    if sys.argv[1:] == ["--day", "3"]:
        llm = LLMClient(LLMConfig(max_retries=5, max_delay_s=30))
        data = encode(model, day3_texts(), llm)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
        out = DAY3 / f"tinycoder_embeddings_day3.{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": model, "dim": data["dim"], "sets": [data]}) + "\n")
        print(f"wrote {out.name}: {len(data['texts'])} texts, vector length {data['dim']}")
        return 0
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
