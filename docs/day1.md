# Day 1: LLM foundations and reliable API calls

**Tuesday, Oct 6, 2026 · 2:00–4:00 PM IST**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/labs/day1_llm_calls.ipynb)

<!-- TODO: Mari — one-paragraph overview of the day -->

## Syllabus topics

1. Topic 1 <!-- TODO: Mari -->
2. Topic 2 <!-- TODO: Mari -->
3. Topic 3 <!-- TODO: Mari -->
4. Topic 4 <!-- TODO: Mari -->

## Capstone progress

A script that sends a paper's abstract to an LLM and returns a plain-language summary.

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
    response = client.generate(prompt=f"Summarize: {abstract}")
    print(response.text)
    ```

=== "Production"

    ```python
    for attempt in range(3):  # (1)!
        try:
            response = client.generate(
                prompt=f"Summarize: {abstract}",
                timeout=30,  # (2)!
            )
            break
        except RateLimitError:
            time.sleep(2 ** attempt)
    print(response.text)
    ```

    1. Retry with exponential backoff on rate limits.
    2. Never wait forever on a network call.

!!! warning "Production"
    <!-- TODO: Mari — hardening notes for Day 1 -->
    Hardening notes coming soon.

!!! failure "What goes wrong in production"
    <!-- TODO: Mari — failure story for Day 1 -->
    Failure story coming soon.

## Wrap-up

- [ ] Notebook saved to my Drive
- [ ] Capstone step for Day 1 working

<!-- TODO: Mari — recap and preview of next session -->
