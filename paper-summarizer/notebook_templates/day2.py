"""Day 2 notebooks: RAG over real PDFs.

Topic order (slides):           where it is taught
  1 Recap of vectors             Demo  D1
  2 What RAG is / I don't know   Demo  D2
  3 Five RAG failure modes       Practice section 3 (each triggered on purpose)
  4 RAG stacks                   Demo  D4
  5 The ingestion pipeline       Practice section 5
  6 The RAG query                Practice section 6

Code cells marked with src(...) are generated from the package, so the notebook shows exactly
the code that the tests run.
"""
from nbgen import Notebook, clean, src

from paper_agent.ingest.splitters import sentence_splitter
from paper_agent.ingest.store import top_k
from paper_agent.rag.answer import PageCitationError, check_page_citations
from paper_agent.rag.naive import NAIVE_PROMPT, naive_answer, naive_chunks, naive_search
from paper_agent.rag.retrieve import keep_confident

INSTALL = "paper-agent[rag] @ git+https://github.com/mari-muthu-k/agent-to-production@main#subdirectory=paper-summarizer"

SETUP = rf'''
import importlib.util
if importlib.util.find_spec("paper_agent") is None:      # Colab: install the course package (about 1 minute)
    !pip install -q "{INSTALL}"

import logging, os, re, time, json
import numpy as np
import openai
from paper_agent.config import load_env
from paper_agent.llm_client import LLMClient, LLMConfig, summarize_calls
from paper_agent.ingest.embeddings import OpenAIEmbedder
from paper_agent.openrouter import make_client, is_openrouter, REASONING_OFF

# Colab Secrets (🔑) or .env: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMBED_MODEL (+ optional LLM_FALLBACK_MODEL)
# OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1, e.g. EMBED_MODEL=nvidia/nemotron-3-embed-1b:free
env = load_env(required=("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "EMBED_MODEL"))
MODEL, EMBED_MODEL = env["LLM_MODEL"], env["EMBED_MODEL"]
PRICE_IN_PER_1M, PRICE_OUT_PER_1M = 0.50, 2.00   # USD per 1M tokens (illustrative: the instructor gives real ones)
PRICE_EMBED_PER_1M = 0.02
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", force=True)

client = make_client(env["LLM_API_KEY"], env["LLM_BASE_URL"])     # OpenAI SDK; works with OpenRouter
REASONING = REASONING_OFF if is_openrouter() else None   # short grounded answers don't need long thinking
llm = LLMClient(LLMConfig(model=MODEL, fallback_model=env.get("LLM_FALLBACK_MODEL") or None,
                          price_in_per_1m=PRICE_IN_PER_1M, price_out_per_1m=PRICE_OUT_PER_1M,
                          max_tokens_param=env.get("LLM_MAX_TOKENS_PARAM") or "max_tokens",
                          reasoning=REASONING), client=client)
embedder = OpenAIEmbedder(EMBED_MODEL, client=client, price_per_1m=PRICE_EMBED_PER_1M)

from paper_agent.papers import fetch_paper, paper_ids, paper_info, edge_case_pdf
PAPERS = {{pid: fetch_paper(pid) for pid in paper_ids()}}      # 4 real papers, CC BY 4.0
'''


# ===========================================================================
# Practice (code-along)
# ===========================================================================
def practice() -> Notebook:
    nb = Notebook("day2-practice", setup=SETUP)
    nb.md("""
        # Day 2 Code-Along: RAG over real PDFs

        **Hands-on AI Workshop · Day 2 · Capstone: Research Paper Summarizer Agent**

        Cell numbers (3.2, 5.4, 6.6 …) match the tags on the slides.

        | Section | What you do |
        |---|---|
        | **3. Five ways RAG fails** | Build a naive RAG in 10 lines, then break it five ways on purpose |
        | **5. The ingestion pipeline** | Load → clean → split → embed → store, done properly (Chroma + numpy) |
        | **6. The RAG query** | Retrieve → threshold → grounded answer with page citations → "I don't know" → eval |
        | **Cheatsheet** | Today on one page |

        **If you fall behind or your runtime restarts:** run the **⏩ catch-up cell** (3.0, 5.0, 6.0) at the start
        of the section you want to rejoin. **If a cell fails:** post the error in the team chat; a TA will help.

        > 📄 Today's papers are **real** arXiv papers, redistributed under CC BY 4.0 (see `MANIFEST.md`):
        > *Mistral 7B*, *Ragas*, *vLLM / PagedAttention* and *Dense X Retrieval*.
        > The small "edge-case" PDFs (two-column, image-only) are **fictional**, generated for the workshop.
    """)

    # -- Section 3 ------------------------------------------------------------------
    nb.section(3, "Five ways RAG fails (on purpose)",
               "First a naive RAG in ten lines. It works on an easy question. Then we break it five ways, "
               "each one fixed later today.")
    nb.md("""
        ### 3.1 Setup

        Colab **Secrets** (🔑, Notebook access ON): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, **`EMBED_MODEL`** (new today).
        The first run installs the course package (about a minute).

        > **OpenRouter:** `LLM_BASE_URL` = `https://openrouter.ai/api/v1`; for embeddings use e.g.
        > `nvidia/nemotron-3-embed-1b:free` or `openai/text-embedding-3-small`. Free (`:free`) models allow
        > 20 requests/minute and 50/day without credits, so today's notebook batches its embedding calls.
    """)
    nb.code(SETUP + '\nfor pid in PAPERS:\n    info = paper_info(pid)\n    print(f"{pid:11} {info[\'pages\']:>2} pages  '
            '{info[\'title\'][:58]}  ({info[\'license\']})")\nprint("✅ ready | chat:", MODEL, "| embeddings:", '
            'EMBED_MODEL)')
    nb.md("### 3.2 A naive RAG in ten lines\n\nFixed 300-character chunks, top-3 by cosine, everything into the prompt.")
    nb.code("from paper_agent.ingest.loaders import load_pdf\n"
            f"NAIVE_PROMPT = {NAIVE_PROMPT!r}\n\n" + src(naive_chunks, naive_search, naive_answer))
    nb.code(r'''
        text = load_pdf(PAPERS["mistral-7b"], layout="naive").text      # read every page straight across
        naive = naive_chunks(text, size=300)
        naive_matrix = embedder.embed(naive)
        question = "How does the rolling buffer cache limit memory use?"
        hits = naive_search(question, naive, naive_matrix, embedder.embed, k=3)
        for score, chunk in hits:
            print(f"{score:.2f}  {chunk[:90]!r}")
        print("\n🤖", naive_answer(question, [c for _, c in hits], client, MODEL))
    ''')
    nb.md("It probably worked. **That's the trap.** Five ways to break it:\n\n### 3.3 💥 Failure 1: bad chunking (and bad parsing before it)")
    nb.code(r'''
        for left, right in list(zip(naive, naive[1:]))[:4]:          # where the 300-char windows cut
            print(f"...{left[-40:]!r}  ✂️  {right[:40]!r}...")
        print("\nRagas page 2, read straight across the page (it has two columns):")
        print("\n".join(load_pdf(PAPERS["ragas"], layout="naive").pages[1].text.splitlines()[:5]))
    ''')
    nb.md("Words, numbers and sentences cut in half; two columns braided together. **Fixed in 5.1–5.3.**\n\n"
          "### 3.4 💥 Failure 2: embedding mismatch")
    nb.code(r'''
        from paper_agent.ingest.embeddings import HashingEmbedder
        other_model = HashingEmbedder(dim=naive_matrix.shape[1])     # a different "embedding model", same size
        same = naive_search(question, naive, naive_matrix, embedder.embed, k=1)[0]
        mixed = naive_search(question, naive, naive_matrix, other_model.embed, k=1)[0]
        print(f"query embedded with the SAME model     : {same[0]:.2f}  {same[1][:70]!r}")
        print(f"query embedded with a DIFFERENT model  : {mixed[0]:.2f}  {mixed[1][:70]!r}")
        print("\nNo error, no warning: just quietly wrong results.")
    ''')
    nb.md("Vectors from different models live in different spaces. **Fixed in 5.5:** stores remember their model and refuse others.\n\n"
          "### 3.5 💥 Failure 3: retrieval noise")
    nb.code(r'''
        for score, chunk in naive_search("What is the best pizza topping?", naive, naive_matrix, embedder.embed, k=3):
            print(f"{score:.2f}  {chunk[:80]!r}")
        print("\nTop-k always returns k chunks, even when none of them is relevant. All go into the prompt.")
    ''')
    nb.md("**Fixed in 6.2:** a score threshold.\n\n### 3.6 💥 Failure 4: context overflow (\"just send everything\")")
    nb.code(r'''
        from paper_agent.tokens import count_tokens
        everything = "\n\n".join(load_pdf(p, layout="naive").text for p in PAPERS.values())
        n = count_tokens(everything)
        print(f"all 4 papers: {n:,} tokens  ->  ${n * PRICE_IN_PER_1M / 1e6:.4f} per question, "
              f"${n * PRICE_IN_PER_1M / 1e6 * 20 * 200:,.2f} for 20 questions x 200 people")
        try:
            r = client.chat.completions.create(model=MODEL, max_tokens=60, messages=[
                {"role": "user", "content": everything + "\n\nQuestion: " + question}])
            print(f"Accepted ({r.usage.prompt_tokens:,} prompt tokens). It fits this model, but you pay for all of it "
                  "on every question, and answers buried mid-context are often missed.")
        except openai.BadRequestError as e:
            print("💥", type(e).__name__, "-", e.message[:170])
    ''')
    nb.md("**Fixed in 6.3:** send only the best chunks that fit a token budget.\n\n### 3.7 💥 Failure 5: hallucination")
    nb.code(r'''
        unanswerable = "Which GPU cloud provider sponsored the training run?"
        hits = naive_search(unanswerable, naive, naive_matrix, embedder.embed, k=3)
        answer = naive_answer(unanswerable, [c for _, c in hits], client, MODEL)
        print("🤖", answer)
        numbers = re.findall(r"\d[\d.,]*\d|\d", answer)
        print("\nnumbers in the answer:", numbers, "| actually in the paper:", [x for x in numbers if x in text])
    ''')
    nb.md("""
        The prompt never said *"you may say you don't know"*, so a confident answer is the path of least resistance.
        **Fixed in 6.4–6.6:** an "insufficient evidence" path, page citations checked against what was retrieved,
        and a number check.

        | # | Failure | Symptom | Fix |
        |---|---|---|---|
        | 1 | Bad chunking / parsing | Answers cut in half, columns mixed | 5.1 column-aware loader, 5.3 sentence-aware splitter |
        | 2 | Embedding mismatch | Plausible but wrong chunks, no error | 5.5 one model per index, enforced |
        | 3 | Retrieval noise | Irrelevant chunks in the prompt | 6.2 score threshold |
        | 4 | Context overflow | 400 error, or cost and buried answers | 6.3 top-k within a token budget |
        | 5 | Hallucination | Confident made-up facts | 6.4–6.6 "I don't know", citation guard, number check |
    """)

    # -- Section 5 ------------------------------------------------------------------
    nb.section(5, "The ingestion pipeline", "Load → clean → split → embed → store. Done once per paper.")
    nb.md("### 5.1 Loaders: read the columns in the right order")
    nb.code(r'''
        from paper_agent.ingest.loaders import load_pdf
        doc = load_pdf(PAPERS["ragas"])                                  # column-aware (the default)
        print(len(doc.pages), "pages | columns per page:", [p.columns for p in doc.pages])
        print("\nRagas page 2, column-aware:\n" + "\n".join(doc.pages[1].text.splitlines()[:5]))
        for name in ("two_column", "image_only"):                       # FICTIONAL edge-case PDFs
            d = load_pdf(edge_case_pdf(name))
            print(f"\n{name}: columns={[p.columns for p in d.pages]} warnings={d.warnings}")
    ''')
    nb.md("An image-only page has no text to extract: we **say so** instead of silently indexing nothing.\n\n"
          "### 5.2 Clean up and find the sections (for citations)")
    nb.code(r'''
        from paper_agent.ingest.cleaning import to_blocks
        blocks = to_blocks(doc.pages)              # drops headers/footers, page numbers, references
        sections = []
        for b in blocks:
            if not sections or sections[-1][0] != b.section:
                sections.append((b.section, b.page))
        print(" · ".join(f"{s} (p.{p})" for s, p in sections))
    ''')
    nb.md("### 5.3 Split: 500–1,000 characters, at sentence boundaries, with overlap (LangChain)")
    nb.code("CHUNK_SIZE, CHUNK_OVERLAP = 800, 120\n\n" + src(sentence_splitter) + "\n\n" + clean(r'''
        sample = blocks[3].text
        for piece in sentence_splitter(chunk_size=300, chunk_overlap=60).split_text(sample)[:3]:
            print(repr(piece), "\n")
    '''))
    nb.code(r'''
        from paper_agent.ingest.splitters import fixed_chunks, sentence_chunks, chunk_stats
        fixed = fixed_chunks(blocks, "ragas", size=300)
        chunks = sentence_chunks(blocks, "ragas")          # the same splitter, per section, tracking pages
        print("fixed 300  :", chunk_stats(fixed))
        print("sentence   :", chunk_stats(chunks))
        c = chunks[4]
        print(f"\n[{c.id}] section={c.section} page={c.page}\n{c.text}")
    ''')
    nb.md("Every chunk now knows its **section and page**: that is what citations will point to.\n\n"
          "### 5.4 Embed")
    nb.code(r'''
        vectors = embedder.embed([c.text for c in chunks])
        print("matrix:", vectors.shape, "| row lengths:", np.round(np.linalg.norm(vectors[:3], axis=1), 3),
              "| tokens embedded so far:", embedder.tokens_used)
    ''')
    nb.md("""
        ### 5.5 Store and search: ✏️ TODO 1 (2 lines)

        With unit-length rows, cosine similarity is a dot product, so one matrix-vector product scores every chunk.
        Return the `k` best `(row_index, score)` pairs, best first.
    """)
    nb.todo(stub=r'''
        def top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list:
            """Indices and scores of the k rows most similar to `query` (all rows L2-normalised)."""
            # ✏️ TODO 1: replace the line below.
            #   1. scores for every row at once: a matrix-vector product
            #   2. the k best indices, highest score first (np.argsort sorts ascending)
            return [(i, float(matrix[i] @ query)) for i in range(k)]
    ''', solution=src(top_k))
    nb.todo_test(r'''
        m = np.eye(4, dtype=np.float32)
        q = np.array([0.1, 0.9, 0.3, 0.0], dtype=np.float32)
        got = [i for i, _ in top_k(m, q, 2)]
        assert got == [1, 2], f"TODO 1: expected rows [1, 2] (best first), got {got}"
        assert len(top_k(m, q, 3)) == 3
        print("✅ TODO 1: top_k returns the most similar rows, best first")
    ''')
    nb.rescue(src(top_k), "`scores = matrix @ query`, then `idx = np.argsort(-scores)[:k]`, "
                          "then return `[(int(i), float(scores[i])) for i in idx]`.")
    nb.code(r'''
        def search(question, k=3):
            q = embedder.embed([question])[0]
            return [(score, chunks[i]) for i, score in top_k(vectors, q, k)]

        for score, c in search("Which three quality aspects of a RAG system does Ragas evaluate?"):
            print(f"{score:.2f}  {c.section} p.{c.page}  {c.text[:80]!r}")
    ''')
    nb.md("That is the whole mechanism of a vector database. Now the same thing in **Chroma**, persisted to disk, "
          "for all four papers. The store remembers its embedding model, which fixes failure 2.")
    nb.code(r'''
        from paper_agent.ingest.pipeline import ingest_pdf
        from paper_agent.ingest.store import ChromaStore
        store = ChromaStore(EMBED_MODEL, path="data/vectorstore")         # one collection per embedding model
        for pid, path in PAPERS.items():
            report, _ = ingest_pdf(path, embedder, store, paper_id=pid)
            print(report)
        print(f"\n{len(store)} chunks in Chroma collection '{store.collection.name}' (saved in data/vectorstore)")
    ''', carry=True)
    nb.code(r'''
        from paper_agent.ingest.embeddings import EmbeddingMismatchError, HashingEmbedder
        try:
            store.search(HashingEmbedder().embed(["rolling buffer cache"])[0], embed_model="hashing-local-256")
        except EmbeddingMismatchError as e:
            print("🛡", e)
    ''')
    nb.md("### 5.6 Fixed vs semantic chunking, side by side\n\nSemantic chunking starts a new chunk where "
          "the topic shifts: where the next sentence's embedding is least similar to the current one.")
    nb.code(r'''
        from paper_agent.ingest.splitters import semantic_chunks
        strategies = {"fixed 300": fixed, "sentence 800": chunks,
                      "semantic": semantic_chunks(blocks, "ragas", embedder.embed)}
        for name, cs in strategies.items():
            s = chunk_stats(cs)
            print(f"{name:13} {s['chunks']:>3} chunks  median {s['median_chars']:>4} chars  "
                  f"{s['cut_mid_sentence']:>2} cut mid-sentence")
        definition = ("Faithfulness refers to the idea that the answer should be grounded in the given context. "
                      "This is important to avoid hallucinations, and to ensure that the retrieved context can act "
                      "as a justification for the generated answer.")
        print("\nWhere does the definition of faithfulness land?")
        for name, cs in strategies.items():
            whole = [c for c in cs if definition in c.text]
            if whole:
                c = whole[0]
                i = c.text.index(definition)
                print(f"[{name}] whole, in one chunk (p.{c.page}): {c.text[i:i + 90]!r}...")
            else:
                print(f"[{name}] ✂️ split across two chunks: neither half answers 'what is faithfulness?'")
    ''')
    nb.md("Semantic chunks follow the argument but cost an embedding per sentence at ingest time. "
          "Sentence-aware fixed-size chunks are the usual default.")

    # -- Section 6 ------------------------------------------------------------------
    nb.section(6, "The RAG query", "Retrieve → keep only confident matches → grounded answer with page "
                                   "citations → verify → or say \"I don't know\".")
    nb.md("### 6.1 Retrieve across all four papers")
    nb.code(r'''
        from paper_agent.rag.retrieve import Retriever, show_hits
        retriever = Retriever(store, embedder, k=5)
        show_hits(retriever.search("How does vLLM share KV cache blocks between parallel samples?"))
    ''', carry=True)
    nb.md("### 6.2 How sure is \"sure\"? Calibrate a score threshold")
    nb.code(r'''
        from paper_agent.rag.evals import golden_questions
        on_topic = [g["question"] for g in golden_questions()]
        off_topic = ["What is the best pizza topping?", "Who won the 2018 football World Cup?", "How do I renew a passport?"]
        embedder.embed(on_topic + off_topic)          # one request for all 13 questions (free tiers count requests)
        best = lambda qs: [retriever.search(q, k=1)[0].score for q in qs]
        on, off = best(on_topic), best(off_topic)
        print("best score, questions the papers answer :", np.round(sorted(on), 2))
        print("best score, questions they don't        :", np.round(sorted(off), 2))
        MIN_SCORE = round((float(np.percentile(on, 25)) + max(off)) / 2, 2)
        print("\nthreshold between the groups -> MIN_SCORE =", MIN_SCORE)
        if min(on) <= max(off):
            print("⚠ the groups overlap: any threshold trades missed answers for noise. Better embeddings separate them.")
    ''', carry=True)
    nb.md("### ✏️ TODO 2 (1 line): drop weak matches")
    nb.todo(stub=r'''
        def keep_confident(hits: list, min_score: float) -> list:
            """Drop hits whose similarity is below the threshold."""
            # ✏️ TODO 2: keep only the hits with h.score >= min_score
            return hits
    ''', solution=src(keep_confident))
    nb.todo_test(r'''
        from paper_agent.ingest.store import Hit
        from paper_agent.ingest.splitters import Chunk
        fake = [Hit(Chunk(f"x:{i}", "x", "text", "results", 1, [1]), s) for i, s in enumerate([0.9, 0.5, 0.2])]
        kept = [h.score for h in keep_confident(fake, 0.4)]
        assert kept == [0.9, 0.5], f"TODO 2: expected [0.9, 0.5], got {kept}"
        assert keep_confident(fake, 0.95) == []
        print("✅ TODO 2: weak matches are dropped")
    ''')
    nb.rescue(src(keep_confident), "`return [h for h in hits if h.score >= min_score]`")
    nb.md("### 6.3 The grounded prompt: chunks as data, (section, page) on every chunk, a way out")
    nb.code(r'''
        from paper_agent.rag.answer import QA_PROMPT, rag_messages, fit_to_budget, allowed_citations
        print(QA_PROMPT)
        print(rag_messages("How was the WikiEval dataset built?",
                           retriever.search("How was the WikiEval dataset built?", k=1))[1]["content"][:400])
    ''')
    nb.md("""
        ### 6.4 Citation guard: ✏️ TODO 3 (1 line)

        Day 1 checked section ids. Now a citation is a **(section, page)** pair, and it must belong to a chunk
        we actually retrieved and sent. `allowed_citations(hits)` gives the set of allowed pairs.
    """)
    nb.todo(stub=r'''
        from paper_agent.schemas import GroundedAnswer

        class PageCitationError(Exception):
            pass

        def check_page_citations(result: GroundedAnswer, hits: list) -> None:
            """Every cited (section, page) must belong to a chunk we actually sent."""
            unknown = set()   # ✏️ TODO 3: the cited (section, page) pairs that are NOT in allowed_citations(hits)
            if unknown:
                raise PageCitationError(f"cites (section, page) pairs that were not retrieved: {sorted(unknown)}")
    ''', solution="from paper_agent.schemas import GroundedAnswer\n\n" + src(PageCitationError, check_page_citations))
    nb.todo_test(r'''
        from paper_agent.ingest.store import Hit
        from paper_agent.ingest.splitters import Chunk
        sent = [Hit(Chunk("p:1", "p", "Recall rose to 74%.", "results", 6, [6]), 0.8)]
        ok = GroundedAnswer(status="answered", answer="Recall rose to 74%.", confidence="high",
                            citations=[{"section": "results", "page": 6}])
        bad = GroundedAnswer(status="answered", answer="Recall rose to 74%.", confidence="high",
                             citations=[{"section": "results", "page": 9}])
        check_page_citations(ok, sent)
        try:
            check_page_citations(bad, sent)
            raise AssertionError("TODO 3: page 9 was never retrieved, but no error was raised")
        except PageCitationError as e:
            print("✅ TODO 3:", e)
    ''')
    nb.rescue("from paper_agent.schemas import GroundedAnswer\n\n" + src(PageCitationError, check_page_citations),
              "`unknown = {(c.section, c.page) for c in result.citations} - allowed_citations(hits)`")
    nb.md("### 6.5 Put it together: `ask()`")
    nb.code(r'''
        from paper_agent.rag.answer import RAGAnswer, NOT_FOUND, unsupported_numbers

        def ask(question, k=5, max_context_tokens=3000):
            hits = keep_confident(retriever.search(question, k=k), MIN_SCORE)
            if not hits:                                    # nothing relevant: say so, and save the call
                return RAGAnswer(question, "insufficient_evidence", NOT_FOUND, llm_called=False)
            hits = fit_to_budget(hits, max_context_tokens)  # best chunks first, within a token budget
            result = llm.chat_structured(rag_messages(question, hits), GroundedAnswer, max_tokens=800)
            check_page_citations(result, hits)              # TODO 3: cite only what was retrieved
            missing = unsupported_numbers(result, hits) if result.status == "answered" else []
            warnings = [f"numbers not found in the cited pages: {missing}"] if missing else []
            return RAGAnswer(question, result.status, result.answer, result.citations, result.confidence, hits, warnings)

        for q in ["How does the rolling buffer cache limit memory use?",
                  "By how much does vLLM improve LLM serving throughput?"]:
            ask(q).show()
    ''', carry=True)
    nb.md("`paper_agent.rag.answer.answer_question` is the same function, plus one repair attempt on a bad "
          "citation (the Day 1 pattern).\n\n### 6.6 \"I don't know\" is a feature")
    nb.code(r'''
        ask("What is the best pizza topping?").show()                                    # stopped by the threshold: no LLM call
        ask("Which GPU cloud provider sponsored the Mistral 7B training run?").show()   # relevant-looking chunks, no answer
        bad = GroundedAnswer(status="answered", answer="It used 512 A100 GPUs.", confidence="high",
                             citations=[{"section": "results", "page": 3}])
        hits = retriever.search("How does the rolling buffer cache limit memory use?", k=3)
        try:
            check_page_citations(bad, hits)
        except PageCitationError as e:
            print("\n🛡 citation guard:", e)
    ''')
    nb.md("### 6.7 Measure retrieval: hit rate @k on 10 golden questions")
    nb.code(r'''
        from paper_agent.rag.evals import hit_rate_at_k
        hit_rate_at_k(retriever, golden_questions(), k=5).show()
        for k in (1, 3):
            print(f"hit rate @{k}: {hit_rate_at_k(retriever, golden_questions(), k=k).hit_rate:.0%}")
    ''')
    nb.md("If the right page never reaches the prompt, no prompt can save the answer. Measure retrieval on its "
          "own, every time you change chunking, embeddings or k.\n\n### 6.8 What did today cost?")
    nb.code(r'''
        import pprint
        pprint.pprint(summarize_calls())
        print(f"embedding tokens: {embedder.tokens_used:,} (≈ ${embedder.tokens_used * PRICE_EMBED_PER_1M / 1e6:.4f})")
    ''')

    # -- Closing --------------------------------------------------------------------
    nb.md("""
        ---
        # RAG Cheatsheet

        | Area | Rules | Practised in |
        |---|---|---|
        | **Parsing** | Check columns, headers and image-only pages · Say what you could not read | 3.3, 5.1 |
        | **Chunking** | 500–1,000 chars at sentence boundaries, with overlap · Keep section + page on every chunk | 5.2, 5.3, 5.6 |
        | **Embeddings** | One model per index, enforced · Re-embed everything when you change model | 3.4, 5.5 |
        | **Retrieval** | Top-k **and** a calibrated threshold · Measure hit rate @k on golden questions | 3.5, 6.2, 6.7 |
        | **Context** | Best chunks first, within a token budget · Never "send the whole paper" | 3.6, 6.3 |
        | **Answers** | Chunks are data, never instructions · Cite (section, page) · Verify citations against what was sent · Let the model say "insufficient evidence" | 3.7, 6.4–6.6 |

        ---
        ### Extensions (homework)

        **E1. Compare strategies with numbers.** Index all four papers with fixed, sentence and semantic chunks and
        compare `hit_rate_at_k` for each.

        **E2. Hybrid search.** Add a keyword score (count of shared words) to the cosine score. Which golden
        questions improve?

        **E3. Re-ranking.** Retrieve 20, then ask the LLM to pick the best 5. What does it cost per question?

        **E4. Your own paper.** Upload a PDF to Colab, `ingest_pdf` it, and write three golden questions for it.

        **Tomorrow (Day 3):** the agent: LangChain tools (`search_paper`, `get_section`, `define_term`, `save_note`),
        an agent loop with limits, and safety guards against instructions hidden inside PDFs.
    """)
    return nb


# ===========================================================================
# Demo (instructor)
# ===========================================================================
def demo() -> Notebook:
    nb = Notebook("day2-demo")
    nb.md("""
        # Day 2 Instructor Demo: vectors, RAG, and RAG stacks (Sections 1, 2, 4)

        **Run by the instructor only** during the talk sections; shared with participants afterwards.
        Cell numbers match the slide tags (D1.1 …). Needs the same secrets as the code-along, including `EMBED_MODEL`.

        **Rehearsal tip:** run D2.1 the day before and note what your model claims without RAG.
    """)
    nb.code(SETUP + '\nprint("✅ ready | chat:", MODEL, "| embeddings:", EMBED_MODEL)')

    nb.md("---\n## D1 Recap: text becomes vectors  ·  *slide: \"Similar meaning, nearby vectors\"*\n\n"
          "### D1.1 Cosine similarity between AI phrases")
    nb.code(r'''
        phrases = ["sliding window attention", "attention over a sliding window", "KV cache memory",
                   "GPU memory used by the KV cache", "pizza with extra cheese"]
        V = embedder.embed(phrases)
        sims = V @ V.T                                  # unit vectors: dot product = cosine similarity
        print(" " * 36 + "".join(f"{i:>6}" for i in range(len(phrases))))
        for i, p in enumerate(phrases):
            print(f"{i} {p[:33]:33}" + "".join(f"{s:6.2f}" for s in sims[i]))
    ''')
    nb.md("Point at: the paraphrases score high, pizza scores near zero. Ask the room: *what would score high that "
          "shouldn't?* (word overlap without shared meaning).\n\n### D1.2 What a vector actually is")
    nb.code(r'''
        v = V[0]
        print("dimensions:", v.shape[0], "| length:", round(float(np.linalg.norm(v)), 3))
        print("first 8 numbers:", np.round(v[:8], 3))
        a, b = V[0], V[1]
        print(f"cosine by hand: {float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)):.3f}  "
              f"vs dot product of unit vectors: {float(a @ b):.3f}")
    ''')

    nb.md("---\n## D2 What RAG is  ·  *slide: \"Retrieve, then answer\"*\n\n### D2.1 Without RAG: ask about a paper the model never saw")
    nb.code(r'''
        q = "In the Ragas paper, how often did Ragas agree with human annotators on faithfulness?"
        r = client.chat.completions.create(model=MODEL, max_tokens=120, temperature=0,
                                           messages=[{"role": "user", "content": q}])
        print("🤖 (no context):", r.choices[0].message.content)
    ''')
    nb.md("### D2.2 With RAG: retrieve, then answer from the pages, with citations")
    nb.code(r'''
        from paper_agent.ingest.pipeline import ingest_pdf
        from paper_agent.ingest.store import NumpyStore
        from paper_agent.rag.retrieve import Retriever, show_hits
        from paper_agent.rag.answer import answer_question
        store = NumpyStore(EMBED_MODEL)
        for pid, path in PAPERS.items():
            ingest_pdf(path, embedder, store, paper_id=pid)
        retriever = Retriever(store, embedder, k=5)
        show_hits(retriever.search(q, k=3))
        answer_question(q, retriever, llm, min_score=0.0).show()
    ''')
    nb.md("### D2.3 Grounding includes saying \"I don't know\"")
    nb.code(r'''
        answer_question("How many GPUs were used to train Mistral 7B?", retriever, llm, min_score=0.0).show()
        answer_question("What is the best pizza topping?", retriever, llm).show()
    ''')

    nb.md("""
        ---
        ## D4 RAG stacks  ·  *slide: "Pick the boring option first"*

        | Layer | Simple / local | Production options | Notes |
        |---|---|---|---|
        | **Vector index** | numpy (exact search) | Chroma, FAISS, pgvector, Qdrant, Weaviate, Pinecone, Azure AI Search | Exact search is fine to ~100k chunks |
        | **Embeddings** | the same endpoint as chat | provider APIs, self-hosted sentence-transformers | One model per index |
        | **Splitters / loaders** | pdfplumber + LangChain splitters | unstructured, cloud document AI | Parsing quality caps everything after it |
        | **Orchestration** | 30 lines of Python | LangChain, LlamaIndex, Haystack | Frameworks save glue code, not thinking |

        ### D4.1 Same query, two stores, same answer
    """)
    nb.code(r'''
        from paper_agent.ingest.store import ChromaStore
        import tempfile
        chroma = ChromaStore(EMBED_MODEL, path=tempfile.mkdtemp())
        chroma.add(store.chunks, store.matrix, embed_model=EMBED_MODEL)
        query = embedder.embed(["How does PagedAttention reduce KV cache waste?"])[0]
        for name, s in [("numpy ", store), ("Chroma", chroma)]:
            t = time.perf_counter()
            hits = s.search(query, k=3)
            print(f"{name}: {(time.perf_counter() - t) * 1000:6.1f} ms  ->", [(h.chunk.id, round(h.score, 3)) for h in hits])
    ''')
    nb.md("Identical results: a vector DB is an index plus persistence, filtering and scale. Start with the "
          "boring option; move when a number (latency, size, ops) tells you to.")
    return nb
