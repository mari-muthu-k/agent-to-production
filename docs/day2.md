# Day 2: Embeddings, retrieval, and RAG

**Wednesday, Oct 7, 2026 · 2:00–4:00 PM IST**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/labs/day2_rag.ipynb)

<!-- TODO: Mari — one-paragraph overview of the day -->

## Syllabus topics

1. Topic 1 <!-- TODO: Mari -->
2. Topic 2 <!-- TODO: Mari -->
3. Topic 3 <!-- TODO: Mari -->
4. Topic 4 <!-- TODO: Mari -->

## Capstone progress

Parse the uploaded PDF, chunk it, embed the chunks, and answer questions with retrieved context.

See the [Capstone](capstone.md) page for how this fits the full app.

## Lab

Open the notebook with the badge above, then **File → Save a copy in Drive** so your
changes are kept.

!!! tip "Pair up"
    One person shares their screen and types; the other reads the instructions and
    reviews. Swap at each lab section.

### Part 1

<!-- TODO: Mari -->

??? success "Solution"
    <!-- TODO: Mari — hint for Part 1 -->
    Hint coming soon.

### Part 2

<!-- TODO: Mari -->

??? success "Solution"
    <!-- TODO: Mari — hint for Part 2 -->
    Hint coming soon.

### Extension challenge

For intermediate participants who finish early. <!-- TODO: Mari -->

## Prototype vs Production

=== "Prototype"

    ```python
    chunks = full_text.split("\n\n")
    context = "\n".join(chunks[:5])
    answer = llm(f"{context}\n\nQ: {question}")
    ```

=== "Production"

    ```python
    chunks = chunk_text(full_text, size=800, overlap=100)  # (1)!
    hits = index.search(embed(question), k=5)
    if not hits or hits[0].score < MIN_SCORE:  # (2)!
        answer = "I couldn't find that in the paper."
    else:
        answer = llm(build_prompt(hits, question))
    ```

    1. Overlapping chunks so sentences aren't cut in half.
    2. Refuse to answer when retrieval finds nothing relevant.

!!! warning "Production"
    <!-- TODO: Mari — hardening notes for Day 2 -->
    Hardening notes coming soon.

!!! failure "What goes wrong in production"
    <!-- TODO: Mari — failure story for Day 2 -->
    Failure story coming soon.

## Wrap-up

- [ ] Notebook saved to my Drive
- [ ] Capstone step for Day 2 working

<!-- TODO: Mari — recap and preview of next session -->
