"""Load a PDF into pages of text (Day 2, section 5.1). Uses pdfplumber (MIT).

Two things go wrong with real papers, and both are handled here:
- Two-column layouts: reading straight across the page interleaves the columns line by line.
  `layout="columns"` (default) detects the gutter and reads the left column, then the right.
- Pages with no text layer (scans, figures saved as images): there is nothing to extract.
  They are kept with empty text and flagged `needs_ocr`, so the ingest report can say so.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

Layout = Literal["columns", "naive"]
# pdfplumber's default (3) glues words together in many LaTeX PDFs ("Mistral7Bisreleased").
TEXT_OPTS = {"x_tolerance": 1.5}


@dataclass
class Page:
    number: int                 # 1-based, as printed in citations
    text: str
    columns: int = 1
    needs_ocr: bool = False     # no extractable text but the page has images


@dataclass
class Document:
    paper_id: str
    title: str
    source: str
    sha256: str
    pages: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def warnings(self) -> list:
        return [f"page {p.number}: no text layer (image-only); needs OCR, skipped" for p in self.pages if p.needs_ocr]


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _two_column_split(page) -> Optional[float]:
    """y coordinate where the two-column body starts, or None if the page is single-column.

    Each text row is classified: 'span' if a word crosses the vertical centre line, otherwise it
    has words on the left, the right, or both. The body starts at the first row after which almost
    no row spans the centre and both halves hold text: that is two columns. Rows above it (title,
    authors) are read across the full width first.
    """
    words = page.extract_words(**TEXT_OPTS)
    mid = page.width / 2
    rows: dict = {}
    for w in words:
        rows.setdefault(round(w["top"] / 3), []).append(w)
    kinds = []            # (top, bottom, spans, has_left, has_right)
    for row in rows.values():
        kinds.append((min(w["top"] for w in row), max(w["bottom"] for w in row),
                      any(w["x0"] < mid - 1 and w["x1"] > mid + 1 for w in row),
                      any(w["x1"] <= mid for w in row), any(w["x0"] >= mid for w in row)))
    kinds.sort()
    for i in range(len(kinds)):
        tail = kinds[i:]
        if len(tail) < 10:
            return None
        spans = sum(k[2] for k in tail)
        left = sum(k[3] for k in tail)
        right = sum(k[4] for k in tail)
        if spans <= max(1, 0.05 * len(tail)) and left >= 5 and right >= 5:
            header = [k[1] for k in kinds[:i]]
            return max(header) + 1 if header else 0.0
    return None


def upright_only(page):
    """Drop rotated characters, e.g. the vertical arXiv stamp in the margin ("5202 rpA ]LC.sc[ ...")."""
    return page.filter(lambda obj: obj.get("object_type") != "char" or obj.get("upright", True))


def extract_page(page, layout: Layout = "columns") -> tuple:
    """(text, number_of_columns) for one pdfplumber page."""
    page = upright_only(page)
    if layout == "naive":
        return page.extract_text(**TEXT_OPTS) or "", 1
    split = _two_column_split(page)
    if split is None:
        return page.extract_text(**TEXT_OPTS) or "", 1
    mid, parts = page.width / 2, []
    if split > 0:
        parts.append(page.crop((0, 0, page.width, split)).extract_text(**TEXT_OPTS) or "")
    parts.append(page.crop((0, split, mid, page.height)).extract_text(**TEXT_OPTS) or "")
    parts.append(page.crop((mid, split, page.width, page.height)).extract_text(**TEXT_OPTS) or "")
    return "\n".join(p for p in parts if p.strip()), 2


def load_pdf(path, paper_id: Optional[str] = None, title: Optional[str] = None,
             layout: Layout = "columns") -> Document:
    import pdfplumber

    path = Path(path)
    doc = Document(paper_id=paper_id or path.stem, title=title or "", source=str(path), sha256=file_sha256(path))
    with pdfplumber.open(path) as pdf:
        if not doc.title:
            doc.title = (pdf.metadata or {}).get("Title") or path.stem
        for i, page in enumerate(pdf.pages, start=1):
            text, columns = extract_page(page, layout)
            needs_ocr = not text.strip() and bool(page.images)
            doc.pages.append(Page(number=i, text=text, columns=columns, needs_ocr=needs_ocr))
    return doc
