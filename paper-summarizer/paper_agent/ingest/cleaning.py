"""Clean extracted text and find section headings (Day 2, section 5.1).

Output is a list of Blocks: runs of text that share one section and one page. Chunks are cut from
blocks later, so every chunk knows its (section, page), which is what citations point to.
"""
import re
from collections import Counter
from dataclasses import dataclass

KNOWN_SECTIONS = ("abstract", "introduction", "background", "related work", "method", "methods", "methodology",
                  "approach", "model", "architecture", "experiments", "experimental setup", "evaluation", "results",
                  "discussion", "analysis", "limitations", "conclusion", "conclusions", "future work",
                  "references", "acknowledgements", "acknowledgments", "appendix")
# "3 Results", "3. Results", "3.2 Ablations", "A Appendix title": a number (or appendix letter),
# then a short capitalised title with real words, nothing else
_NUMBERED = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?|[A-H]\.?(?:\d{1,2})?)"
                       r"\s+([A-Z][A-Za-z][A-Za-z0-9 ,:&()'/-]{1,70})$")
SKIP_SECTIONS = ("references", "acknowledgements", "acknowledgments")


@dataclass
class Block:
    section: str
    page: int
    text: str


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "body"


def heading(line: str):
    """The section name if `line` looks like a heading, else None."""
    line = line.strip()
    if not 3 <= len(line) <= 80:
        return None
    bare = re.sub(r"^[\dA-Z](?:\.\d+)*\.?\s+", "", line) if re.match(r"^[\dA-Z](?:\.\d+)*\.?\s", line) else line
    if bare.lower().rstrip(".:") in KNOWN_SECTIONS:
        return slug(bare.rstrip(".:"))
    m = _NUMBERED.match(line)
    if not m or re.search(r"[.!?,;]$", line) or re.search(r"\d{3,}", line):
        return None
    title = m.group(1)
    words = title.split()
    long_words = [w for w in words if len(w) >= 4 and w[0].isalpha()]
    # Headings are short, made of real words; figure labels ("Block 1 Years b") and fragments are not.
    stray = [w for w in words if len(w) == 1 and w.lower() != "a" or w.isdigit()]
    if not 1 <= len(words) <= 8 or not long_words or stray:
        return None
    return slug(title)


def repeated_lines(pages: list) -> set:
    """Lines that repeat on most pages (running headers/footers, arXiv stamps)."""
    if len(pages) < 3:
        return set()
    counts = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
        counts.update(set(lines[:2] + lines[-2:]))
    return {ln for ln, n in counts.items() if n >= max(3, len(pages) // 2)}


def clean_line_breaks(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # de-hyphenate words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()          # paragraphs become one line


def to_blocks(pages: list, skip_sections=SKIP_SECTIONS) -> list:
    """Pages -> Blocks with section names. Text before the first heading is 'front-matter'."""
    noise = repeated_lines(pages)
    blocks, section = [], "front-matter"

    def flush(buf: list, section: str, page_number: int) -> None:
        text = clean_line_breaks("\n".join(buf))
        if text and section not in skip_sections:
            blocks.append(Block(section, page_number, text))
        buf.clear()

    for page in pages:
        buf: list = []
        for line in page.text.splitlines():
            s = line.strip()
            if not s or s in noise or re.fullmatch(r"\d{1,3}", s):   # blank, header/footer, page number
                continue
            name = heading(s)
            if name:
                flush(buf, section, page.number)
                section = name
                continue
            buf.append(s)
        flush(buf, section, page.number)
    return blocks
