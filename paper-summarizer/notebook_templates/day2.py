"""Day 2 notebooks: from PDF to cited answers. Follows reference/Day2_Instructor_Guide.docx cell for cell.

Practice (code-along)   Sections 4-8, cells 4.0-8.5, four TODOs; tags match the slides
Demo (instructor)       D2.1-D2.4: toy vectors, cross product, real embeddings

Every function cell is generated from paper_agent/rag/pdf_rag.py with src(...) and every constant with
consts(...), so the notebook runs exactly the code the tests run. In Colab nothing is installed from
the course package: cell 4.0 installs three pinned libraries and downloads llm_client.py and the
day's files.
"""
from pathlib import Path

from nbgen import Notebook, clean, consts, src

from paper_agent.rag import pdf_rag as r

DAY2 = Path(__file__).resolve().parents[2] / "notebooks" / "day2"
REPO = "https://raw.githubusercontent.com/mari-muthu-k/agent-to-production/main"
PINS = "pypdf==6.19.0 chromadb==1.5.9 langchain-text-splitters==1.1.3"
FILES = {"llm_client.py": "paper-summarizer/paper_agent/llm_client.py",
         "tinycoder.pdf": "notebooks/day2/tinycoder.pdf",
         "fake.pdf": "notebooks/day2/fake.pdf",
         "sixty_pages.pdf": "notebooks/day2/sixty_pages.pdf",
         **{p.name: f"notebooks/day2/{p.name}" for p in sorted(DAY2.glob("tinycoder_embeddings*.json"))}}

FETCH = rf'''
import importlib, os, shutil, urllib.request
from pathlib import Path
REPO = "{REPO}"
FILES = {FILES!r}

def fetch(name, force=False):
    """Copy a course file next to the notebook: from the repo checkout (Docker), else from GitHub (Colab)."""
    if Path(name).exists() and not force:
        return
    local = Path(os.environ.get("COURSE_REPO", "/nonexistent")) / FILES[name]
    if local.exists():
        shutil.copy(local, name)
    else:
        urllib.request.urlretrieve(f"{{REPO}}/{{FILES[name]}}", name)
'''

PROVIDER = r'''
# Your provider, from Colab Secrets (🔑, notebook access ON) or environment variables. Table at the top.
# ⚠️ Never put company data into free APIs.
import json, logging, re, hashlib, base64
from typing import Optional
import numpy as np
NAMES = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMBED_MODEL", "LLM_FALLBACK_MODEL", "OTHER_EMBED_MODEL",
         "LLM_REASONING_EFFORT")
try:
    from google.colab import userdata
    for name in NAMES:
        try:
            if userdata.get(name):
                os.environ[name] = userdata.get(name)
        except Exception:                      # secret not defined, or notebook access is off
            pass
except ImportError:
    pass
missing = [n for n in NAMES[:4] if not os.environ.get(n)]
if missing:
    raise RuntimeError(f"Missing {missing}: add them in Colab Secrets (🔑) and turn on notebook access. "
                       "The table at the top says where to find each one for your provider.")
import llm_client
if not hasattr(llm_client.LLMClient, "embed"):   # an older (Day 1) llm_client.py was already in this runtime
    fetch("llm_client.py", force=True)
    importlib.reload(llm_client)
from llm_client import LLMClient, LLMConfig, CALL_LOG, summarize_calls
logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
logging.getLogger("llm_client").setLevel(logging.INFO)          # one JSON log line per call
llm = LLMClient(LLMConfig(max_retries=5, max_delay_s=30))       # free tiers: wait up to 30 s on a 429
EMBED_MODEL = os.environ["EMBED_MODEL"]
'''

SETUP = (f'''
# 4.0 SETUP: installs three libraries (pinned), downloads today's files, connects to your provider.
import importlib.util
if any(importlib.util.find_spec(m) is None for m in ("pypdf", "chromadb", "langchain_text_splitters")):
    !pip install -q {PINS}
''' + FETCH + '''
for name in FILES:
    fetch(name)
''' + PROVIDER + '''
from pydantic import BaseModel, model_validator
PDF = Path("tinycoder.pdf")
USE_PRECOMPUTED = globals().get("USE_PRECOMPUTED", False)       # set it to True in 6.0 if embeddings are down
''')

PRECOMPUTED_SWITCH = ("USE_PRECOMPUTED = False   # ← True if the embedding API is down: search, Chroma and "
                      "Section 8 still run from precomputed vectors")


def stub(solution: str, answer: str, todo: str) -> str:
    """The TODO version of a function: its answer line replaced by a TODO comment and a working-but-wrong line."""
    if answer not in solution:
        raise ValueError(f"answer line not found: {answer!r}")
    at = solution.index(answer)
    indent = solution[solution.rfind("\n", 0, at) + 1:at]
    return solution.replace(answer, todo.replace("\n", "\n" + indent))


def cell(*parts: str) -> str:
    """One code cell from several pieces (generated source, constants, hand-written lines)."""
    return "\n\n\n".join(clean(p) for p in parts if p.strip())


# ===========================================================================
# Practice (code-along)
# ===========================================================================
def practice() -> Notebook:
    nb = Notebook("day2-practice", setup=SETUP)
    nb.md("""
        # Day 2 Code-Along: from PDF to cited answers

        **Hands-on AI Workshop · Day 2 · Capstone: Research Paper Summarizer Agent**

        Today the agent stops being handed the paper and starts reading it:
        **PDF → parse → chunk → embed → vector search → answer with citations.**
        Cell numbers (4.1, 5.3, 7.4b …) match the tags on the slides.

        | Section | What you build |
        |---|---|
        | **4. Parse the PDF** | Validate the upload (TODO 1), parse page by page, clean |
        | **5. Chunk** | Fixed-size vs recursive splitting (TODO 2), page and section metadata |
        | **6. Embed and vector search** | Embed once and cache, cosine search by hand (TODO 3), Chroma, the model guard |
        | **7. Answer with citations** | Grounded prompt, `GroundedAnswer`, `ask()`, "I don't know" (TODO 4) |
        | **8. When RAG fails** | Break it four ways, then measure retrieval |

        **Fell behind, or your runtime restarted?** Run the **⏩ catch-up cell** (5.0, 6.0, 7.0, 8.0) at the start
        of the section you want to rejoin. **A cell failed?** Post the error in the Q&A channel; a TA will help.

        ### Your provider: any OpenAI-compatible API with a free tier and an embedding model

        Add these in **Colab Secrets** (🔑 in the left sidebar), and turn on **notebook access** for each.
        Model names change often: use the ones pinned in the workshop channel.

        | Secret | What it is | Google Gemini | OpenRouter |
        |---|---|---|---|
        | `LLM_BASE_URL` | The API's address | `https://generativelanguage.googleapis.com/v1beta/openai` | `https://openrouter.ai/api/v1` |
        | `LLM_API_KEY` | Your key | Google AI Studio → **Get API key** | openrouter.ai → **Keys** (starts `sk-or-`) |
        | `LLM_MODEL` | The chat model | A Gemini chat model from the AI Studio model list | A model whose id ends in `:free` (openrouter.ai/models, price filter: free) |
        | `LLM_FALLBACK_MODEL` | Used when the first model is busy or out of its daily quota | A second Gemini chat model | A second `:free` model |
        | `EMBED_MODEL` | The embedding model: one per index | A Gemini model whose name contains `embedding` | An embedding model (openrouter.ai/models, filter: embeddings) |
        | `OTHER_EMBED_MODEL` *(optional)* | Cell 8.4 only: a second embedding model with the same vector length | A second Gemini embedding model | A second embedding model |

        > ⚠️ **Never put company data into free APIs.** Free tiers may log, keep or train on what you send. Today
        > we use a fictional paper; for 7.6, use a public paper (for example from arXiv), not an internal document.
        >
        > Free tiers also cap requests per minute and per day. `llm_client` retries the per-minute limit for you;
        > a **daily** cap says so plainly: switch `LLM_MODEL` to your fallback or use another key.

        > 📄 **tinycoder.pdf is FICTIONAL**: the Day 1 TinyCoder paper, rebuilt as a six-page PDF for this workshop.
    """)

    # -- Section 4 -------------------------------------------------------------------------
    nb.section(4, "Parse the PDF", "Validate the upload, parse page by page, clean.", catch_up=False)
    nb.md("### 4.0 Setup\n\nInstalls pypdf, chromadb and LangChain's text splitters (pinned versions), and downloads "
          "tinycoder.pdf, the 4.3b test files and llm_client.py. About two minutes the first time.")
    nb.code(SETUP)
    nb.md("### 4.1 Setup check: one chat call, one embedding call\n\nYou should see **ready**, your two model names, "
          "and your **vector length**: it depends on your embedding model. Green tick in the chat when you see it.")
    nb.code(r'''
        # 4.1
        reply = llm.chat([{"role": "user", "content": "Reply with one word: ready"}], max_tokens=200)
        vector = llm.embed(["TinyCoder uses sliding-window attention."])[0]
        print("✅ ready" if reply else "⚠️ the chat model returned no text", f"(the model said {(reply or '').strip()[:30]!r})")
        print("chat model     :", llm.models()[0])
        print("embedding model:", EMBED_MODEL)
        print("vector length  :", len(vector))
    ''')
    nb.md("### 4.2 Meet tinycoder.pdf\n\nYesterday's TinyCoder paper, now a six-page PDF with the mess real PDFs "
          "have. Open it from the Files panel (📁) and scroll through it.")
    nb.code(r'''
        # 4.2
        from pypdf import PdfReader
        print(f"{PDF.name}: {PDF.stat().st_size / 1024:.0f} KB, {len(PdfReader(PDF).pages)} pages")
    ''')
    nb.md("""
        ### 4.3a Never trust an upload: ✏️ TODO 1 (one line)

        `validate_upload` checks the first bytes are `%PDF` (a file name can lie), the size limit, and encryption.
        **TODO 1:** if the PDF has more than `MAX_PAGES` pages, raise an `UploadError` that says how many it has and
        what the limit is.
    """)
    answer1 = 'if len(reader.pages) > MAX_PAGES: raise UploadError(f"{len(reader.pages)} pages; the limit is {MAX_PAGES}")'
    head1 = consts(r, "PAPER_ID", "MAX_PAGES", "MAX_BYTES") + "\n\n\n"
    sol1 = head1 + src(r.UploadError, r.validate_upload)
    nb.todo(stub=stub(sol1, answer1, "# ✏️ TODO 1: if the PDF has more than MAX_PAGES pages, raise an UploadError\n"
                                     "#   that says how many pages it has and what the limit is."),
            solution=sol1)
    nb.md("### 4.3b Test it: two rejections, one pass\n\nIf all three pass, your TODO isn't saved: re-run 4.3a.")
    nb.todo_test(r'''
        # 4.3b
        results = {}
        for name in ("fake.pdf", "sixty_pages.pdf", "tinycoder.pdf"):
            try:
                validate_upload(name)
                results[name] = "pass"
                print(f"✅ {name}: accepted")
            except UploadError as e:
                results[name] = "rejected"
                print(f"🛑 {name}: rejected: {e}")
        assert results == {"fake.pdf": "rejected", "sixty_pages.pdf": "rejected", "tinycoder.pdf": "pass"}, \
            "TODO 1: sixty_pages.pdf should be rejected (60 pages; the limit is 50)"
        print("✅ TODO 1: two rejections, one pass")
    ''')
    nb.rescue(sol1, f"`{answer1}`")
    nb.md("### 4.4 Parse page by page\n\nWe keep the page number from the very first step: that's what makes page "
          "citations possible later. Look at raw page 4: the header, the footer, `atten-` / `tion`, and Table 1.")
    nb.code(src(r.parse_pages) + "\n\n\npages = parse_pages(PDF)", carry=True)
    nb.code(r'''
        # 4.4
        print(len(pages), "pages; each one is", list(pages[0]))
        print("\n----- raw page 4, as pypdf extracts it -----")
        print(pages[3]["text"])
    ''')
    nb.md("### 4.5 Clean the pages\n\nRemove headers and footers, re-join hyphenated words, stop at References, and "
          "refuse a PDF with almost no text (a scanned image).")
    nb.code(consts(r, "PAGE_NUMBER", "HEADING", "REFERENCES", "MIN_DOCUMENT_CHARS") + "\n\n\n"
            + src(r.is_heading, r.boilerplate_lines, r.clean_page, r.clean_pages) + "\n\n\nclean = clean_pages(pages)",
            carry=True)
    nb.code(r'''
        # 4.5
        print("removed as header/footer:", sorted(boilerplate_lines(pages)))
        print(f"kept {len(clean)} of {len(pages)} pages (everything from References on is dropped)")
        print("\n----- page 4, cleaned -----")
        print(clean[3]["text"])
    ''')
    nb.md("### 4.6 What the parser saw that you didn't")
    nb.code(r'''
        # 4.6
        for p in pages:
            for line in p["text"].splitlines():
                if "AI tools summarizing" in line:
                    print(f"page {p['page']}: {line}")
        print("\nWhite text: invisible in the PDF viewer, fully visible to the parser, and still there after cleaning.")
    ''')

    # -- Section 5 -------------------------------------------------------------------------
    nb.section(5, "Chunk", "Why chunk: precision (one vector per idea) and budget (three chunks cost far less than "
                           "three pages). The usual range is 500 to 1,000 characters; we use 800.")
    nb.md("### 5.2 Fixed-size chunking: cut every 800 characters")
    nb.code(cell(src(r.fixed_chunks), r'''
        # 5.2
        text = "\n\n".join(p["text"] for p in clean)
        fixed = fixed_chunks(text, 800)
        print(len(fixed), "chunks of 800 characters")
        for i in range(len(fixed) - 1):
            if "(pass@1)" in fixed[i] and "42% for the 7B" not in fixed[i]:
                print(f"\nchunk {i + 1} ends   : ...{fixed[i][-45:]!r}")
                print(f"chunk {i + 2} starts : {fixed[i + 1][:45]!r}...")
                print("\n✂️ The key result is cut in half.")
    '''))
    nb.md("""
        ### 5.3 The recursive splitter: ✏️ TODO 2 (one line)

        LangChain's `RecursiveCharacterTextSplitter`: `chunk_size` 800, `chunk_overlap` 150, and `separators` in
        priority order: a blank line, a line break, a full stop and space, a space, nothing.
    """)
    answer2 = ('splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, '
               'separators=["\\n\\n", "\\n", ". ", " ", ""])')
    sol2 = src(r.make_splitter)
    nb.todo(stub=stub(sol2, answer2, "# ✏️ TODO 2: chunk_size 800, chunk_overlap 150, and separators in priority order:\n"
                                     "#   blank line, line break, full stop + space, space, nothing.\n"
                                     'splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0, '
                                     'separators=[""])   # replace this fixed-size cut'),
            solution=sol2)
    nb.todo_test(r'''
        # 5.3
        splitter = make_splitter()
        settings = (splitter._chunk_size, splitter._chunk_overlap, splitter._separators)
        assert settings == (800, 150, ["\n\n", "\n", ". ", " ", ""]), f"TODO 2: got {settings}"
        whole = [c for c in splitter.split_text(text) if "(pass@1) versus 42% for the 7B baseline" in c]
        assert whole, "TODO 2: the pass@1 sentence is still cut"
        print("The pass@1 sentence, whole, in one chunk:\n", whole[0][:220], "...")
        print("\n✅ TODO 2: chunk_size 800, overlap 150, paragraph → line → sentence → word → character")
    ''')
    nb.rescue(sol2, f"`{answer2}`")
    nb.md("### 5.4 Metadata: page and section\n\nSplit page by page and track the latest heading, so every chunk "
          "gets a page, a section and an ID like `p4-c2`. Those IDs are what the model will cite.")
    nb.code(consts(r, "SEPARATORS") + "\n\n\n" + src(r.chunk_pages) + "\n\n\nCHUNKS = chunk_pages(clean)", carry=True)
    nb.code(r'''
        # 5.4
        print(f"{'id':7} {'page':>4}  {'section':26} {'chars':>5}  starts with")
        for c in CHUNKS:
            print(f"{c['id']:7} {c['page']:>4}  {c['section'][:26]:26} {len(c['text']):>5}  {c['text'][:38]!r}")
        print(f"\n{len(CHUNKS)} chunks")
    ''')
    nb.md("### 5.5 Semantic chunking (watch)\n\nThe instructor sets `RUN_SEMANTIC = True`; leave it False and watch. "
          "It costs one embedding per sentence.")
    nb.code(cell("# 5.5\nRUN_SEMANTIC = False", consts(r, "SENTENCE_END"), src(r.semantic_chunks), r'''
        n_sentences = len(SENTENCE_END.split(" ".join(text.split())))
        if RUN_SEMANTIC:
            semantic = semantic_chunks(clean[3]["text"])        # page 4: Results
            for s in semantic:
                print(f"[{len(s)} chars] {s[:150]}...\n")
            print(f"{len(semantic)} semantic chunks for page 4. Cost: one embedding per sentence "
                  f"({n_sentences} for the whole paper); the recursive splitter needed none.")
        else:
            print(f"Skipped: semantic chunking would embed all {n_sentences} sentences of the paper. Watch the screen.")
    '''))

    # -- Section 6 -------------------------------------------------------------------------
    nb.section(6, "Embed and vector search", "Embed once, search by hand, then store it properly.",
               preamble=PRECOMPUTED_SWITCH)
    nb.md("### 6.1 Embed one chunk\n\nThrough llm_client's new `embed()`, so embedding calls get yesterday's retries, "
          "breaker, budget guard and log line. The log shows input tokens when your provider reports them; some "
          "don't, and then it says `null`.")
    nb.code(consts(r, "K", "LONG_FILES", "ATTENTION", "LEARNING_RATE", "IPL", "JAVASCRIPT", "CONCLUSIONS", "NOISE_QUERY",
                   "EVAL_QUESTIONS", "PRECOMPUTED") + "\n\n\n" + src(r.embed_texts, r.load_precomputed), carry=True)
    nb.code(r'''
        # 6.1
        chunk = next(c for c in CHUNKS if c["id"] == "p2-c1")
        vector = embed_texts([chunk["text"]])[0]
        print(f"{chunk['id']} ({chunk['section']}) → {len(vector)} numbers, length {np.linalg.norm(vector):.3f}")
        print("first five:", np.round(vector[:5], 4))
    ''')
    nb.md("### 6.2 Embed every chunk, once\n\nOne batched call for every chunk. **Run it twice**: the second time "
          "prints `cache hit` and makes no call, because we fingerprint the PDF's bytes.")
    nb.code(src(r.file_sha256, r.embed_chunks), carry=True)
    nb.code("# 6.2\nVECTORS = embed_chunks(CHUNKS, PDF)", carry=True)
    nb.code(r'''
        last = [c for c in CALL_LOG if c.kind == "embed"][-1:]
        if last and last[0].usage_reported:
            print(f"last embedding call: {last[0].input_tokens:,} input tokens, ${last[0].cost_usd:.6f}")
        elif last:
            print("last embedding call: your provider didn't report token counts (input_tokens: null)")
        print("matrix:", VECTORS.shape, "(one row per chunk)")
    ''')
    nb.md("""
        ### 6.3 Vector search by hand: ✏️ TODO 3 (one line)

        `M` is every chunk vector, one row per chunk; `q` is the question's vector. TODO 3 is cosine similarity for
        every chunk at once: `M @ q`, divided by each row's length times `q`'s length. Not every provider returns
        unit-length vectors, so we always divide.
    """)
    answer3 = "scores = (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))"
    sol3 = src(r.cosine_scores, r.top_k)
    nb.todo(stub=stub(sol3, answer3, "# ✏️ TODO 3: cosine similarity for every row: M @ q, divided by each row's\n"
                                     "#   length (np.linalg.norm(M, axis=1)) times q's length (np.linalg.norm(q)).\n"
                                     "scores = M @ q   # a bare dot product: wrong when vectors aren't unit length"),
            solution=sol3)
    nb.todo_test(r'''
        # 6.3 test: vectors that are NOT unit length
        M_test = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
        q_test = np.array([0.0, 0.5])
        got = cosine_scores(M_test, q_test)
        assert np.allclose(got, [0.8, 0.0, 1.0]), f"TODO 3: expected [0.8, 0.0, 1.0], got {np.round(got, 3)}"
        print("✅ TODO 3: cosine similarity divides by both lengths")
    ''')
    nb.rescue(sol3, f"`{answer3}`")
    nb.code(r'''
        # 6.3
        M, NP_MODEL = VECTORS, EMBED_MODEL                   # the numpy index, and the model that built it
        q = embed_texts([LONG_FILES])[0]
        print(f"{LONG_FILES!r}\n")
        for i, score in top_k(M, q, 3):
            print(f"  {score:.3f}  {CHUNKS[i]['id']:6} {CHUNKS[i]['section']:14} {CHUNKS[i]['text'][:60]!r}")
    ''', carry=True)
    nb.md("### 6.4 Store it in Chroma\n\nVectors, text and metadata together, saved to a folder. Cosine "
          "**distance** = 1 − similarity: lower is closer. `reset_index()` empties it if you ever see a dimension "
          "or \"already exists\" error.")
    nb.code(src(r.open_index, r.reset_index, r.add_to_index, r.index_model, r.search_chroma)
            + '\n\n\ncollection = open_index("index")\nadd_to_index(collection, CHUNKS, VECTORS)', carry=True)
    nb.code(r'''
        # 6.4
        print(f"{collection.count()} chunks in Chroma collection '{collection.name}', saved in ./index\n")
        for h in search_chroma(collection, q, paper_id=PAPER_ID):
            print(f"  distance {h['distance']:.3f}  (similarity {h['score']:.3f})  {h['id']:6} {h['section']}")
        print("\nSame top three as NumPy, shown as distances.")
    ''')
    nb.md("### 6.5 Your turn: query the index\n\nChange the question, run it, and post your best and worst query "
          "in the chat. Ideas: *what hardware did they train on?*, *is the benchmark trustworthy?*, *why is it faster?*")
    nb.code(r'''
        # 6.5
        MY_QUESTION = "What hardware did they train on?"
        for h in search_chroma(collection, embed_texts([MY_QUESTION])[0], paper_id=PAPER_ID):
            print(f"  {h['score']:.3f}  p. {h['page']}  {h['id']:6} {h['text'][:80]!r}")
    ''')
    nb.md("### 6.6 Guard the embedding model\n\nAn index only works with the model that built it. Check before "
          "every search.")
    nb.code(src(r.IndexMismatchError, r.check_model, r.search_numpy, r.retrieve), carry=True)
    nb.code(r'''
        # 6.6
        print("this index was built with:", index_model(collection))
        try:
            check_model(index_model(collection), "some-other-embedding-model")
        except IndexMismatchError as e:
            print("🛡", e)
    ''')

    # -- Section 7 -------------------------------------------------------------------------
    nb.section(7, "Answer with citations", "Augment the prompt with what we found, generate, and check citations.",
               preamble=PRECOMPUTED_SWITCH)
    nb.md("### 7.1 The grounded prompt\n\nEach chunk inside a tag with its ID and page, then four rules.")
    nb.code(consts(r, "RULES") + "\n\n\n" + src(r.build_prompt), carry=True)
    nb.code(r'''
        # 7.1
        messages = build_prompt(LONG_FILES, retrieve(LONG_FILES))
        print(messages[0]["content"], "\n")
        print(messages[1]["content"][:900], "...")
    ''')
    nb.md("### 7.2 The GroundedAnswer schema\n\n`answer`, `found` and `citations`, with one rule built in: "
          "found=True needs a citation.")
    nb.code(src(r.GroundedAnswer) + "\n\n\n" + consts(r, "NOT_FOUND"), carry=True)
    nb.code(r'''
        # 7.2
        try:
            GroundedAnswer(answer="It uses sliding-window attention.", found=True, citations=[])
        except Exception as e:
            print("🛡 rejected:", e.errors()[0]["msg"])
    ''')
    nb.md("### 7.3 ask(): end to end\n\nCheck the model, embed the question, retrieve the top three, build the "
          "prompt, call llm_client (validated, one repair), check citations against what was retrieved, add pages.")
    nb.code(consts(r, "MAX_ANSWER_TOKENS", "MIN_SCORE") + "\n\n\n" + src(r.CitationError, r.check_citations, r.answer_from)
            + "\n\n\ndef stop_early(best_score, MIN_SCORE):\n    return None       # layer 1 arrives in 7.4 (TODO 4)"
            + "\n\n\n" + src(r.ask), carry=True)
    nb.code(r'''
        # 7.3
        n = len(CALL_LOG)
        print(ask(ATTENTION))
        print("\ncalls:", [(c.kind, c.model, c.input_tokens) for c in CALL_LOG[n:]])
    ''')
    nb.md("### 7.4a How relevant is the best chunk?\n\nThe best score for three questions: one the paper answers, "
          "one it doesn't, one off-topic. `MIN_SCORE` goes between the off-topic score and the on-topic ones.")
    nb.code(r'''
        # 7.4a
        best = {question: retrieve(question)[0]["score"] for question in (LONG_FILES, LEARNING_RATE, IPL)}
        for question, score in best.items():
            print(f"  {score:.3f}  {question}")
        MIN_SCORE = round((best[IPL] + best[LEARNING_RATE]) / 2, 2)    # halfway: set it from YOUR numbers
        print("\nMIN_SCORE =", MIN_SCORE)
    ''', carry=True)
    nb.md("### ✏️ TODO 4 (one line): stop before the model when nothing is relevant")
    answer4 = "if best_score < MIN_SCORE: return NOT_FOUND"
    sol4 = src(r.stop_early)
    nb.todo(stub=stub(sol4, answer4, "# ✏️ TODO 4: if best_score is below MIN_SCORE, return NOT_FOUND (no model call)"),
            solution=sol4)
    nb.todo_test(r'''
        # 7.4 test
        assert stop_early(0.10, 0.50) == NOT_FOUND, "TODO 4: a best score below MIN_SCORE should return NOT_FOUND"
        assert stop_early(0.90, 0.50) is None, "TODO 4: a best score above MIN_SCORE should carry on (return None)"
        print("✅ TODO 4: below MIN_SCORE, answer NOT_FOUND without calling the model")
    ''')
    nb.rescue(sol4, f"`{answer4}`")
    nb.md("### 7.4b Three layers of \"I don't know\"")
    nb.code(r'''
        # 7.4b
        for question in (IPL, LEARNING_RATE, JAVASCRIPT):
            n = len(CALL_LOG)
            result = ask(question)
            chats = sum(c.kind == "chat" for c in CALL_LOG[n:])
            print(f"❓ {question}\n   {result}\n   chat calls: {chats}\n")
    ''')
    nb.md("### 7.5 The hidden instruction, again\n\nWe ask for the conclusions **and** the limitations. Did the "
          "answer call TinyCoder better than all large models, or leave the limitations out? Post what you got.")
    nb.code(r'''
        # 7.5
        hits = retrieve(CONCLUSIONS)
        print("retrieved:", [h["id"] for h in hits], "| hidden line among them:",
              any("AI tools summarizing" in h["text"] for h in hits))
        result = ask(CONCLUSIONS)
        print("\n", result)
        text = result.answer.lower()
        print("\nsays 'better than all large models':", "better than all large models" in text)
        print("mentions the limitations           :",
              any(w in text for w in ("limitation", "python only", "javascript", "not evaluated", "benchmark")))
    ''')
    nb.md("### 7.6 Your turn: your own paper\n\nUpload a public PDF under 50 pages through the Files panel (📁), put "
          "its name below, and run. Or stay with TinyCoder. ⚠️ No company documents: this goes to a free API.")
    nb.code(src(r.ingest), carry=True)
    nb.code(r'''
        # 7.6
        MY_PDF = "tinycoder.pdf"                 # ← your uploaded file's name
        MY_PAPER_ID = Path(MY_PDF).stem + "-mine"
        my_chunks = ingest(MY_PDF, paper_id=MY_PAPER_ID)
        print(ask("What is the main contribution of this paper?", paper_id=MY_PAPER_ID))
    ''')

    # -- Section 8 -------------------------------------------------------------------------
    nb.section(8, "When RAG fails", "", preamble=PRECOMPUTED_SWITCH)
    nb.md("""
        ### 8.1 Five common ways RAG fails

        | Failure | Demo | What you'll see | Defence |
        |---|---|---|---|
        | Bad chunking | 8.2 | 120-character chunks split 31% from 39% | Recursive splitter, overlap |
        | Retrieval noise | 8.3 | References outrank the methods chunk | Clean pages, filter, small k |
        | Embedding mismatch | 8.4 | Another model: no error, scores near zero | Store and check the model name |
        | Context overflow | 8.4 | k = 20: thousands of tokens, a worse answer | Top 3, a token budget |
        | Hallucination | – | A claim no chunk supports | Grounded prompt, found flag, citation guard |

        ### 8.2 Bad chunking: 120 characters, no overlap
    """)
    nb.code(r'''
        # 8.2
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        tiny = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0, separators=SEPARATORS)
        small_chunks = chunk_pages(clean, tiny)
        small_index = open_index("index", "papers_chunk120")
        add_to_index(small_index, small_chunks, embed_chunks(small_chunks, PDF))
        for h in search_chroma(small_index, embed_texts([LONG_FILES])[0]):
            print(f"  {h['score']:.3f}  {h['id']:7} {h['text']!r}")
        saved, collection = collection, small_index          # ask() reads `collection`
        try:
            print("\n", ask(LONG_FILES))
        finally:
            collection = saved
        print("\nNothing crashed. The answer just got worse.")
    ''')
    nb.md("### 8.3 Retrieval noise: no cleaning")
    nb.code(r'''
        # 8.3
        raw_chunks = chunk_pages(pages)                      # pages straight from pypdf: headers, footers, References
        noisy_index = open_index("index", "papers_noclean")
        add_to_index(noisy_index, raw_chunks, embed_chunks(raw_chunks, PDF))
        query = embed_texts([NOISE_QUERY])[0]
        for name, index in (("without cleaning", noisy_index), ("with cleaning", collection)):
            print(f"{NOISE_QUERY!r}, {name}:")
            for h in search_chroma(index, query):
                print(f"  {h['score']:.3f}  p. {h['page']}  {' '.join(h['text'].split())[:70]!r}")
            print()
    ''')
    nb.md("### 8.4 Embedding mismatch, then context overflow")
    nb.code(r'''
        # 8.4a: query with a different embedding model (no guard)
        q_ours = embed_texts([LONG_FILES])[0]
        OTHER = os.environ.get("OTHER_EMBED_MODEL")
        q_other = embed_texts([LONG_FILES], model=OTHER)[0] if OTHER else None
        if q_other is None or len(q_other) != len(q_ours):
            if q_other is not None:
                print(f"{OTHER} returns {len(q_other)} numbers, not {len(q_ours)}: Chroma would raise a dimension error.")
            rng = np.random.default_rng(0)                   # simulate a second model: same length, different space
            q_other = q_ours[rng.permutation(len(q_ours))] * rng.choice([-1.0, 1.0], len(q_ours))
            OTHER = "a simulated second model"
        print(f"query embedded with {OTHER}: {len(q_other)} numbers, the same length as the index, so no error\n")
        hits_other = search_chroma(collection, q_other)
        for h in hits_other:
            print(f"  {h['score']:+.3f}  {h['id']:6} {h['text'][:60]!r}")
        best_other = hits_other[0]["score"]
        print(f"\nscores collapse to around zero. Best {best_other:+.3f} vs MIN_SCORE {MIN_SCORE}:",
              "the threshold catches it →", stop_early(best_other, MIN_SCORE) or "it would NOT catch it")
        print("\nwithout the threshold, the model answers from the wrong chunks:")
        print(answer_from(LONG_FILES, hits_other))
    ''')
    nb.code(r'''
        # 8.4b: context overflow
        for k in (3, 20):
            hits_k = retrieve(LONG_FILES, k=k)
            n = len(CALL_LOG)
            result = answer_from(LONG_FILES, hits_k)
            chat = [c for c in CALL_LOG[n:] if c.kind == "chat"][-1]
            prompt_chars = sum(len(m["content"]) for m in build_prompt(LONG_FILES, hits_k))
            tokens = f"{chat.input_tokens:,} input tokens" if chat.usage_reported else "tokens not reported"
            print(f"k={k:>2}: {len(hits_k)} chunks, {prompt_chars:,} characters, {tokens}\n   {result}\n")
    ''')
    nb.md("### 8.5 Measure retrieval, don't guess\n\nSix questions with the page that answers them: is that page "
          "in the top three? Change one setting, run again, and watch the number move.")
    nb.code(cell(src(r.hit_rate), r'''
        # 8.5: change one setting, run again
        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, CLEAN = 800, 150, 3, True

        if (CHUNK_SIZE, CHUNK_OVERLAP, CLEAN) == (800, 150, True):
            paper = PAPER_ID                                 # the index from 6.4
        else:
            splitter_85 = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                         separators=SEPARATORS)
            chunks_85 = chunk_pages(clean if CLEAN else pages, splitter_85)
            paper = f"tinycoder-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{CLEAN}"
            add_to_index(collection, chunks_85, embed_chunks(chunks_85, PDF), paper)
        rows = hit_rate(EVAL_QUESTIONS, paper, k=TOP_K)
        for question, page, got, hit in rows:
            print(f"{'✅' if hit else '❌'} p. {page}  top {TOP_K}: {got}  {question}")
        print(f"\nhit@{TOP_K}: {sum(hit for *_, hit in rows)}/{len(rows)}")
    '''))
    nb.code(r'''
        # What did today cost?
        import pprint
        pprint.pprint(summarize_calls())
    ''')

    # -- Closing ---------------------------------------------------------------------------
    nb.md("""
        ---
        # Production Cheatsheet: RAG

        - **Uploads and parsing:** validate before parsing; keep page numbers from step one; strip headers, footers and references; reject PDFs with no text.
        - **Chunking:** 500–1,000 characters on paragraphs, then sentences; 10–20% overlap; page, section and ID on every chunk; semantic only when measured.
        - **Embedding:** one model per index, recorded; re-embed when it changes; batch calls and cache by file hash; same reliable client.
        - **Retrieval:** dense for meaning, sparse for exact terms, hybrid for both; cosine similarity; know whether your store reports distance; filter by paper and user; keep k at 3–5; calibrate the not-found threshold.
        - **Answering:** only retrieved chunks, in tags; cite chunk IDs and show pages; validate citations against what was retrieved; "doesn't cover this" is a correct answer.
        - **Test for:** bad chunking, embedding mismatch, retrieval noise, context overflow, hallucination, with a small question set before and after every change.

        ---
        ### Extensions (homework)

        **E1. Hybrid search.** Add sparse (keyword) search next to dense search: score every chunk with BM25 over
        its words, rank the chunks by BM25 and by cosine, and merge the two rankings with **reciprocal rank
        fusion**: each chunk scores `1 / (60 + rank_bm25) + 1 / (60 + rank_dense)`. Then ask a question with an
        exact identifier, like *"Llama-3.1-8B"* or *"HTTP 429"*, and compare the two rankings with the fused one.
        Re-run 8.5: did hit@3 change?

        **E2. A reranker.** Retrieve the top 20, then ask the chat model to pick the best 3. What does it cost per
        question, and does 8.5 improve?

        **E3. Your own paper.** Run `ingest()` and `ask()` on a public paper from your field, and write three
        questions with expected pages for it, like 8.5.

        **E4. Tables and scans.** Try a layout-aware parser (Docling, Unstructured) on Table 1, or OCR on a scanned
        page. What changes in 4.4?

        ---
        ### Tomorrow (Day 3): an agent with tools, in LangChain

        Our search becomes a **tool** the model decides when to call; the **agent** plans, calls tools and knows
        when it's done; **safety guards** stop the hidden line on page 5; and **memory** remembers who it's
        explaining to. Tonight: finish anything you missed with the catch-up cells, and try `ingest()` and `ask()`
        on a paper from your field.
    """)
    return nb


# ===========================================================================
# Demo (instructor): D2.1-D2.4
# ===========================================================================
DEMO_SETUP = FETCH + '\nfetch("llm_client.py")\n' + PROVIDER


def demo() -> Notebook:
    nb = Notebook("day2-demo")
    nb.md("""
        # Day 2 Instructor Demo: vectors and similarity (D2.1–D2.4)

        **Run by the instructor only** (Slides 7–9); shared with participants afterwards. Needs the same Secrets as the
        code-along. ⚠️ Never put company data into free APIs.

        **Before the session:** run D2.3 and D2.4 once and write the real scores on a sticky note.
    """)
    nb.code("# Setup: downloads llm_client.py and connects to your provider\n" + clean(DEMO_SETUP))
    nb.md("---\n## D2.1 A vector you can read · *Slides 7–8*\n\nThree made-up dimensions: "
          "**[about GPUs, about databases, about security]**.")
    nb.code(r'''
        # D2.1
        V = {"A": ("CUDA kernel optimisation", np.array([9, 1, 1])),
             "B": ("GPU memory bandwidth", np.array([8, 2, 1])),
             "C": ("SQL index tuning", np.array([1, 9, 2])),
             "D": ("SQL injection attack", np.array([1, 6, 9])),
             "G": ("Long survey paragraph", np.array([12, 10, 10]))}

        def cosine(a, b):
            """Dot product divided by both lengths: direction only."""
            return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

        try:
            import matplotlib.pyplot as plt
            ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
            for name, (label, v) in V.items():
                ax.quiver(0, 0, 0, *v, arrow_length_ratio=0.08)
                ax.text(*v, f" {name}: {label}")
            ax.set_xlabel("about GPUs"); ax.set_ylabel("about databases"); ax.set_zlabel("about security")
            ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.set_zlim(0, 12)
            plt.show()
        except ImportError:
            print("(matplotlib isn't installed here, so no drawing; Colab has it)")

        print("magnitude (length): square, add, square root")
        for name, (label, v) in V.items():
            print(f"  |{name}| = {np.linalg.norm(v):5.2f}   {v}  {label}")
    ''')
    nb.code(r'''
        # D2.1 continued: dot product, then cosine similarity
        A, B, C, D, G = (V[k][1] for k in "ABCDG")
        print("dot product: multiply position by position, then add")
        for x, y in (("A", "B"), ("A", "C"), ("A", "G")):
            print(f"  {x} · {y} = {V[x][1] @ V[y][1]}")
        print("\ncosine similarity: the dot product divided by both lengths")
        for x, y in (("A", "B"), ("A", "C"), ("C", "D"), ("A", "G")):
            print(f"  cos({x}, {y}) = {cosine(V[x][1], V[y][1]):.2f}")
        print("\nA · G (128) beats A · B (75), but cos(A, G) 0.76 < cos(A, B) 0.99: the focused match wins.")
    ''')
    nb.md("---\n## D2.2 Cross product, and why we don't use it · *Slide 8*")
    nb.code(r'''
        # D2.2
        print("A × B =", np.cross(A, B), "(a new vector at right angles to both: 3D graphics, not search)")
        real = np.array(llm.embed(["How do I speed up a CUDA kernel?", "Tips for optimising GPU kernels"]))
        print(f"\ntwo real embeddings from {EMBED_MODEL}: {real.shape[1]} numbers each")
        try:
            np.cross(real[0], real[1])
        except ValueError as e:
            print("np.cross on them →", type(e).__name__ + ":", e)
    ''')
    nb.md("---\n## D2.3 Real embeddings, real scores · *Slide 9*\n\n**Ask first:** which pair scores highest? "
          "Is sentence 1 closer to 2 or to 3?")
    nb.code(r'''
        # D2.3: one embedding
        one = np.array(llm.embed(["How do I speed up a CUDA kernel?"])[0])
        print(f"{EMBED_MODEL}: {len(one)} numbers, length {np.linalg.norm(one):.3f}")
        print("first eight:", np.round(one[:8], 4))
    ''')
    nb.code(r'''
        # D2.3: six sentences, three of them with "kernel" in three senses
        SIX = ["How do I speed up a CUDA kernel?",
               "Tips for optimising GPU kernels",
               "How the Linux kernel schedules processes",
               "Choosing a kernel for an SVM classifier",
               "My SQL query is slow; should I add an index?",
               "How do I prevent SQL injection?"]
        E = np.array(llm.embed(SIX))
        print(f"{len(SIX)} vectors of {E.shape[1]} numbers\n")
        S = (E @ E.T) / np.outer(np.linalg.norm(E, axis=1), np.linalg.norm(E, axis=1))    # cosine, every pair
        print("     " + "".join(f"{j + 1:>6}" for j in range(len(SIX))))
        for i, row in enumerate(S):
            print(f"{i + 1:>3}  " + "".join(f"{x:6.2f}" for x in row) + f"   {SIX[i]}")
        pairs = [(S[i, j], i + 1, j + 1) for i in range(len(SIX)) for j in range(i + 1, len(SIX))]
        top = max(pairs)
        print(f"\nhighest pair: {top[1]} and {top[2]} ({top[0]:.2f}); 1 vs 2: {S[0, 1]:.2f}, 1 vs 3: {S[0, 2]:.2f}")
    ''')
    nb.md("---\n## D2.4 Exact identifiers · *Slide 9*")
    nb.code(r'''
        # D2.4
        llama = np.array(llm.embed(["Llama-3.1-8B", "Llama-3.1-70B"]))
        print(f"{llama.shape[1]} numbers each")
        print(f"cos(Llama-3.1-8B, Llama-3.1-70B) = {cosine(llama[0], llama[1]):.3f}")
        print("Very similar vectors, very different models: exact identifiers need keyword (sparse) search or filters.")
    ''')
    return nb
