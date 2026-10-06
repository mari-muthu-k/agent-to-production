# Day 2: From PDF to cited answers

**Wednesday, Oct 7, 2026 · 2:00–4:00 PM IST**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/notebooks/day2/Practice.ipynb)

Today the agent stops being handed the paper and starts reading it:
**PDF → parse → chunk → embed → vector search → answer with citations.** First we look at what RAG is
and the vector maths underneath it (dot product, cosine similarity, and why not the cross product);
from 2:35 you build it. By 4 PM you have the TinyCoder paper indexed on disk, and an `ask()` function
that answers with page numbers and says "the paper doesn't cover this" when it doesn't.

## Syllabus topics

1. What RAG is, where it's used, and how it works: an ingestion pipeline and a query pipeline
2. Vectors, dot product and cosine similarity; dense vs sparse search
3. Parsing and cleaning a PDF, and chunking with boundaries and overlap
4. Embeddings, vector search by hand, and a vector store (Chroma)
5. Grounded answers with citations, and three layers of "I don't know"
6. Five ways RAG fails, and measuring retrieval

## Capstone progress

Parse the uploaded PDF, chunk it, embed the chunks, and answer questions with retrieved context and page
citations.

See the [Capstone](capstone.md) page for how this fits the full app.

## Before you start

Day 2 needs one more Colab Secret than Day 1: **`EMBED_MODEL`**, an embedding model from your provider.
The [setup page](setup.md#3-get-an-llm-api-key) has the table for Gemini and OpenRouter. Use the model
names pinned in the workshop channel.

!!! danger "No company data"
    Never put company data into free APIs. Today's paper, `tinycoder.pdf`, is **fictional**, written for
    this workshop. For your own paper in 7.6, use a public one.

## Lab

Open the notebook with the badge above, then **File → Save a copy in Drive** so your changes are kept.
Each section starts with a **⏩ catch-up cell** (5.0–8.0): if you fall behind or your runtime restarts,
run it and you're level again.

!!! tip "Pair up"
    One person shares their screen and types; the other reads the instructions and
    reviews. Swap at each lab section.

### Section 4: Parse the PDF (TODO 1)

Validate the upload before doing anything expensive with it: the first bytes must be `%PDF`, the file
under 20 MB, not encrypted, and (your TODO) no more than 50 pages.

??? success "Solution"
    `if len(reader.pages) > MAX_PAGES: raise UploadError(f"{len(reader.pages)} pages; the limit is {MAX_PAGES}")`

### Section 5: Chunk (TODO 2)

Cutting every 800 characters splits the key result in half. LangChain's recursive splitter cuts at
paragraphs, then lines, then sentences, with overlap.

??? success "Solution"
    `splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""])`

### Section 6: Embed and vector search (TODO 3)

Vector search is cosine similarity plus sorting. Not every provider returns unit-length vectors, so
divide by the lengths.

??? success "Solution"
    `scores = (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))`

### Section 7: Answer with citations (TODO 4)

If even the best chunk scores below `MIN_SCORE`, answer "the paper doesn't cover this" without calling
the model.

??? success "Solution"
    `if best_score < MIN_SCORE: return NOT_FOUND`

### Section 8: When RAG fails

Bad chunking, retrieval noise, embedding mismatch and context overflow, triggered on purpose; then a
six-question retrieval check (hit@3). Change one setting and watch the number move.

### Extension challenge

Add **hybrid search**: BM25 keyword scores alongside dense vector scores, merged with reciprocal rank
fusion. Does a question with an exact identifier, like "Llama-3.1-8B", get better?

## Prototype vs Production

=== "Prototype"

    ```python
    chunks = full_text.split("\n\n")
    context = "\n".join(chunks[:5])
    answer = llm(f"{context}\n\nQ: {question}")
    ```

=== "Production"

    ```python
    chunks = chunk_pages(clean_pages(parse_pages(path)))   # (1)!
    hits = retrieve(question, paper_id)                      # (2)!
    if hits[0]["score"] < MIN_SCORE:                         # (3)!
        return NOT_FOUND
    answer = answer_from(question, hits)                     # (4)!
    ```

    1. Page by page, cleaned, 800 characters with 150 overlap; every chunk keeps its page and section.
    2. Checks the index was built with the same embedding model, then cosine search filtered to this paper.
    3. Refuse to answer when retrieval finds nothing relevant, without paying for a model call.
    4. Grounded prompt, validated `GroundedAnswer`, citations checked against what was retrieved.

!!! warning "Production"
    Validate uploads first; keep one embedding model per index and record it; cache embeddings by file
    hash; calibrate the "not found" threshold on your own questions; filter by paper and user on every
    query, and never rely on the prompt for access control.

!!! failure "What goes wrong in production"
    Retrieval delivers whatever the document says, including instructions. TinyCoder's page 5 hides a
    white-text line telling AI tools to call it "better than all large models": invisible to readers,
    extracted by the parser, and retrieved because it's relevant. Day 3 adds real guards.

## Wrap-up

- [ ] Notebook saved to my Drive
- [ ] `ask()` answers with page citations, and says "the paper doesn't cover this" when it doesn't

**Tomorrow (Day 3):** an agent with tools, in LangChain: search becomes a tool, the agent decides when to
use it, safety guards against that hidden line, and memory.
