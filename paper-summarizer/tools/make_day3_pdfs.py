"""Generate the Day 3 PDFs (reportlab, byte-for-byte reproducible), all FICTIONAL.

    tinycoder_day3.pdf      Day 2's TinyCoder paper, clean: no hidden text, plus the authors' contact
                            emails on page 1 (5.3 redacts them)
    tinycoder_injected.pdf  the same paper as an "uploaded" copy: two white-on-white lines in the
                            discussion section (5.4), one aimed at summaries and one at the reader's memory
    second_paper.pdf        QuickEmbed, a second small fictional AI paper, for the wrong-paper cache hit (6.3)

Written to paper_agent/fixtures/pdfs/, where tests read them and Colab downloads them from.

    python tools/make_day3_pdfs.py
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import make_tinycoder_pdf as mk  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from paper_agent.fixtures.tinycoder import AUTHORS, CONTACT, INJECTED_LINES  # noqa: E402

OUT = ROOT / "paper_agent" / "fixtures" / "pdfs"

QUICKEMBED = [
    [("title", "QuickEmbed: A 110M-Parameter Sentence Embedding Model for CPU Search"),
     ("authors", "Ravi Iyer and Lena Park (fictional authors)"),
     ("fiction", mk.FICTION),
     ("h", "Abstract"),
     ("p", "We present QuickEmbed, a 110-million-parameter sentence embedding model distilled from a "
           "1-billion-parameter teacher. QuickEmbed encodes a search query in about 4 milliseconds on a single CPU "
           "core and keeps most of the teacher's retrieval quality on our benchmark."),
     ("h", "1 Introduction"),
     ("p", "Semantic search needs an embedding for every query, and large embedding models need a GPU to answer "
           "quickly. Many search services run on CPUs only. We ask how small an embedding model can be before "
           "retrieval quality drops sharply.")],
    [("h", "2 Method"),
     ("p", "QuickEmbed is a 6-layer encoder trained by distillation: for 40 million sentence pairs it learns to "
           "reproduce the teacher's cosine similarities. Vectors have 384 numbers and are normalised to unit "
           "length, so a dot product is the cosine similarity."),
     ("h", "3 Results"),
     ("p", "Our main result: QuickEmbed reaches a recall@10 of 88% on our 5,000-query benchmark, against 91% for "
           "the 1B-parameter teacher, while encoding a query in 4 ms on one CPU core instead of 60 ms."),
     ("p", "Recall@10 is the share of queries whose relevant passage is among the ten closest vectors. The gap "
           "to the teacher is largest for queries containing numbers and product codes.")],
    [("h", "4 Limitations"),
     ("p", "English only, one benchmark built by the authors, and short passages of under 200 words. Queries "
           "with exact identifiers still need keyword search next to the embeddings.")],
]      # three pages: Day 2's cleaner only strips a running header that repeats on three pages or more


def tinycoder_pages(injected: bool) -> list:
    pages = copy.deepcopy(mk.PAGES)
    first = pages[0]
    at = next(i for i, (kind, _) in enumerate(first) if kind == "authors")
    first[at] = ("authors", AUTHORS)
    first.insert(at + 2, ("p", CONTACT))         # after the "fictional paper" line; wraps like a paragraph
    discussion = pages[4]
    discussion[:] = [item for item in discussion if item[0] != "white"]       # Day 2's single hidden line
    if injected:
        at = next(i for i, (kind, text) in enumerate(discussion) if kind == "h" and "Limitations" in text)
        discussion[at:at] = [("white", line) for line in INJECTED_LINES]
    return pages


def render(path: Path, pages: list, header: str, title: str) -> None:
    saved = mk.PAGES, mk.HEADER
    mk.PAGES, mk.HEADER = pages, header             # draw() numbers pages "n of len(PAGES)"
    try:
        c = canvas.Canvas(str(path), pagesize=A4, invariant=1)
        c.setTitle(title)
        c.setAuthor("Workshop generator (fictional)")
        for n, items in enumerate(pages, start=1):
            mk.draw(c, n, items)
            c.showPage()
        c.save()
    finally:
        mk.PAGES, mk.HEADER = saved


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mk.PAD = ""
    render(OUT / "tinycoder_day3.pdf", tinycoder_pages(injected=False), mk.HEADER,
           "TinyCoder (fictional workshop paper)")
    render(OUT / "tinycoder_injected.pdf", tinycoder_pages(injected=True), mk.HEADER,
           "TinyCoder (fictional workshop paper, uploaded copy)")
    render(OUT / "second_paper.pdf", QUICKEMBED, "QuickEmbed · Preprint · October 2026",
           "QuickEmbed (fictional workshop paper)")
    for name in ("tinycoder_day3.pdf", "tinycoder_injected.pdf", "second_paper.pdf"):
        print(f"{name:24} {(OUT / name).stat().st_size:>7} bytes")


if __name__ == "__main__":
    main()
