"""Split blocks into chunks (Day 2, sections 3.1 and 5.2).

fixed_chunks     every N characters, no regard for words or sentences (the naive baseline)
sentence_chunks  LangChain RecursiveCharacterTextSplitter: 500-1,000 characters, breaks at paragraph,
                 then sentence, then word boundaries, with overlap
semantic_chunks  break where the topic shifts: adjacent-sentence embedding similarity drops

Chunks never cross sections (a section heading is a natural topic boundary) but may cross pages;
`page` is where the chunk starts and `pages` lists every page it touches.
"""
import re
from dataclasses import dataclass, field

import numpy as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK = 500
MAX_CHUNK = 1000


@dataclass
class Chunk:
    id: str
    paper_id: str
    text: str
    section: str
    page: int
    pages: list = field(default_factory=list)

    def tag(self) -> str:
        """How the chunk is shown to the model: data inside delimiters, with its citation handle."""
        return f'<chunk id="{self.id}" section="{self.section}" page="{self.page}">{self.text}</chunk>'


def _sections(blocks: list) -> list:
    """Group consecutive blocks of the same section: [(section, text, [(offset, page), ...])]."""
    groups = []
    for b in blocks:
        if groups and groups[-1][0] == b.section:
            sec, text, marks = groups[-1]
            marks.append((len(text) + 1, b.page))
            groups[-1] = (sec, text + " " + b.text, marks)
        else:
            groups.append((b.section, b.text, [(0, b.page)]))
    return groups


def _pages_for(start: int, end: int, marks: list) -> list:
    pages = [p for off, p in marks if off <= start][-1:] + [p for off, p in marks if start < off < end]
    return sorted(set(pages)) or [marks[0][1]]


def _make_chunks(paper_id: str, pieces: list) -> list:
    """pieces: [(section, text, start, end, marks)] -> numbered Chunks."""
    out = []
    for section, text, start, end, marks in pieces:
        pages = _pages_for(start, end, marks)
        out.append(Chunk(f"{paper_id}:{len(out):04d}", paper_id, text.strip(), section, pages[0], pages))
    return [c for c in out if c.text]


def fixed_chunks(blocks: list, paper_id: str, size: int = 300, overlap: int = 0) -> list:
    """Cut every `size` characters. Splits words, numbers and sentences in half."""
    pieces = []
    for section, text, marks in _sections(blocks):
        step = max(size - overlap, 1)
        for start in range(0, len(text), step):
            pieces.append((section, text[start:start + size], start, min(start + size, len(text)), marks))
    return _make_chunks(paper_id, pieces)


def sentence_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator="end",          # the full stop stays with its sentence
        add_start_index=True,
    )


def sentence_chunks(blocks: list, paper_id: str, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = sentence_splitter(chunk_size, chunk_overlap)
    pieces = []
    for section, text, marks in _sections(blocks):
        for doc in splitter.create_documents([text]):
            start = doc.metadata["start_index"]
            pieces.append((section, doc.page_content, start, start + len(doc.page_content), marks))
    return _make_chunks(paper_id, pieces)


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list:
    """[(start, sentence)] using a simple rule: . ! ? followed by space and a capital or digit."""
    out, pos = [], 0
    for part in _SENTENCE.split(text):
        start = text.find(part, pos)
        out.append((start, part))
        pos = start + len(part)
    return [(s, p) for s, p in out if p.strip()]


def semantic_chunks(blocks: list, paper_id: str, embed, percentile: float = 80,
                    min_chars: int = MIN_CHUNK, max_chars: int = MAX_CHUNK) -> list:
    """Start a new chunk where the next sentence is least similar to the current one.

    `embed(list_of_texts) -> np.ndarray` of L2-normalised rows. Breakpoints are the adjacent-sentence
    similarity drops below the given percentile of dissimilarity; chunks are kept within
    [min_chars, max_chars] by merging small groups and cutting oversized ones at a sentence.
    """
    pieces = []
    for section, text, marks in _sections(blocks):
        sents = split_sentences(text)
        if len(sents) < 3:
            pieces.append((section, text, 0, len(text), marks))
            continue
        vecs = np.asarray(embed([s for _, s in sents]), dtype=np.float32)
        distance = 1 - np.sum(vecs[:-1] * vecs[1:], axis=1)          # between sentence i and i+1
        cut_after = set(np.where(distance >= np.percentile(distance, percentile))[0].tolist())
        start, length = sents[0][0], 0
        for i, (s_start, sent) in enumerate(sents):
            length = s_start + len(sent) - start
            last = i == len(sents) - 1
            too_big = not last and length + len(sents[i + 1][1]) > max_chars
            if last or too_big or (i in cut_after and length >= min_chars):
                end = s_start + len(sent)
                pieces.append((section, text[start:end], start, end, marks))
                if not last:
                    start = sents[i + 1][0]
    return _make_chunks(paper_id, pieces)


def chunk_stats(chunks: list) -> dict:
    lengths = [len(c.text) for c in chunks] or [0]
    return {"chunks": len(chunks), "min_chars": min(lengths), "median_chars": int(np.median(lengths)),
            "max_chars": max(lengths),
            # starts lower-case or ends without punctuation: the chunk was cut mid-sentence
            "cut_mid_sentence": sum(1 for c in chunks if c.text and (c.text[0].islower() or c.text[-1].isalnum()))}
