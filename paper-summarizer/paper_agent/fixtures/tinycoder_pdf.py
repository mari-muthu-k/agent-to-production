"""TinyCoder as a real three-page PDF (Day 4, cell 4.0). FICTIONAL: written for this workshop.

The same paper as Days 1-3, shortened to three pages: no hidden text, no email addresses. reportlab with
invariant=1, so the bytes (and the paper id, a hash of them) are the same on every run and machine.
It never mentions a learning rate or who funded the work: the golden set's two refusal questions (7.1).
"""
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

W, H = A4
MARGIN, SIZE, LEADING = 64, 10, 13.5
HEADER = "TinyCoder · Preprint · October 2026"
PAGES = [
    [("title", "TinyCoder: A 1.3B-Parameter Code Model with Sliding-Window Attention"),
     ("authors", "Asha Raman, Daniel Okafor and Mei Lin (fictional authors)"),
     ("fiction", "FICTIONAL PAPER, written for this workshop"),
     ("h", "Abstract"),
     ("p", "We present TinyCoder, a 1.3-billion-parameter language model for Python code completion. Using "
           "sliding-window attention and 200 billion tokens of filtered Python code, TinyCoder comes within one "
           "point of a 7-billion-parameter baseline on our completion benchmark while running about 4x faster on "
           "a single GPU. We also report where the small model falls short."),
     ("h", "1 Introduction"),
     ("p", "Large code models are accurate but expensive to serve. Every completion request in an editor has to "
           "return in a fraction of a second, and a 7B-parameter model needs a data-centre GPU to do that for many "
           "users at once. Small models are cheap to run but usually weaker."),
     ("p", "We ask whether architecture and data choices can close the gap for a single programming language. "
           "Our answer is a narrow model: Python only, trained on carefully filtered code, with an attention "
           "pattern chosen for speed.")],
    [("h", "2 Methods"),
     ("p", "As its attention mechanism, TinyCoder uses sliding-window attention in every layer, with a 1,024-token "
           "window: each token attends only to the 1,024 tokens before it. We chose it for cost: with full "
           "attention the compute grows quadratically with the length of the file, while with a fixed window it "
           "grows linearly."),
     ("p", "The rest of the model is a plain decoder-only transformer: 24 layers, a hidden size of 2,048 and 16 "
           "attention heads, for 1.3 billion parameters in total. It uses the same tokenizer as the baseline, so "
           "the two models see identical inputs."),
     ("h", "3 Training and evaluation"),
     ("p", "TinyCoder was trained on 200B tokens of permissively licensed Python from public repositories. "
           "Training ran on 64 GPUs for 9 days. The data was deduplicated, and code that looked auto-generated "
           "was removed."),
     ("p", "We evaluate on 2,000 held-out Python functions written by the authors. For each function the model "
           "sees the signature and the docstring and must write the body. We report pass@1: the share of "
           "functions whose first generated body passes all of its unit tests.")],
    [("h", "4 Results"),
     ("p", "TinyCoder solves 41% of functions on the first attempt (pass@1) versus 42% for the 7B baseline, "
           "with about 4x faster inference on the same GPU: 2,400 tokens per second against 600."),
     ("p", "Quality is not uniform across file sizes. On files longer than 1,024 tokens it drops to 31% versus "
           "39% for the baseline, because code that depends on definitions outside the window is harder to "
           "complete."),
     ("h", "5 Discussion"),
     ("p", "Small, specialised models can be competitive for single-language completion. For an editor plugin, a "
           "model that is four times faster and one point less accurate is often the better trade."),
     ("h", "6 Limitations"),
     ("p", "Python only: other languages, such as JavaScript and Java, were not evaluated. One benchmark, built "
           "by the authors, so the results may favour the style of code we trained on. Overlap between the "
           "benchmark and the training data was checked by exact match only.")],
]


def make_pdf(path=None) -> bytes:
    """The three-page TinyCoder PDF as bytes (and written to `path` if given). Same bytes on every run."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4, invariant=1)          # invariant: no timestamps, stable bytes
    c.setTitle("TinyCoder (fictional workshop paper)")
    c.setAuthor("Workshop generator (fictional)")
    for number, items in enumerate(PAGES, start=1):
        c.setFont("Helvetica", 8)
        c.drawCentredString(W / 2, H - 36, HEADER)
        c.drawCentredString(W / 2, 30, f"Page {number} of {len(PAGES)}")
        y = H - 80
        for kind, text in items:
            if kind == "title":
                c.setFont("Helvetica-Bold", 15)
                for line in simpleSplit(text, "Helvetica-Bold", 15, W - 2 * MARGIN):
                    c.drawCentredString(W / 2, y, line)
                    y -= 19
            elif kind in ("authors", "fiction"):
                c.setFont("Helvetica" if kind == "authors" else "Helvetica-Oblique", 10 if kind == "authors" else 9)
                c.drawCentredString(W / 2, y, text)
                y -= 15 if kind == "authors" else 28
            elif kind == "h":
                c.setFont("Helvetica-Bold", 11.5)
                c.drawString(MARGIN, y - 4, text)
                y -= LEADING + 8
            else:
                c.setFont("Helvetica", SIZE)
                for line in simpleSplit(text, "Helvetica", SIZE, W - 2 * MARGIN):
                    c.drawString(MARGIN, y, line)
                    y -= LEADING
                y -= 8
        c.showPage()
    c.save()
    data = buffer.getvalue()
    if path:
        with open(path, "wb") as f:
            f.write(data)
    return data
