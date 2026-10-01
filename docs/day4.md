# Day 4: Shipping and operating

**Friday, Oct 9, 2026 · 2:00–4:00 PM IST**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/labs/day4_shipping.ipynb)

<!-- TODO: Mari — one-paragraph overview of the day -->

## Syllabus topics

1. Topic 1 <!-- TODO: Mari -->
2. Topic 2 <!-- TODO: Mari -->
3. Topic 3 <!-- TODO: Mari -->
4. Topic 4 <!-- TODO: Mari -->

## Capstone progress

Wrap the agent in a small web app, add logging and evaluation, and share a public demo.

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
    @app.post("/summarize")
    def summarize(file):
        return agent.run(file.read())
    ```

=== "Production"

    ```python
    @app.post("/summarize")
    def summarize(file):
        if file.size > MAX_BYTES:  # (1)!
            raise HTTPException(413, "PDF too large")
        with log_span("summarize", paper=file.name):  # (2)!
            return agent.run(file.read())
    ```

    1. Reject oversized uploads before they reach the model.
    2. Log every request so you can debug and measure cost.

!!! warning "Production"
    <!-- TODO: Mari — hardening notes for Day 4 -->
    Hardening notes coming soon.

!!! failure "What goes wrong in production"
    <!-- TODO: Mari — failure story for Day 4 -->
    Failure story coming soon.

## Wrap-up

- [ ] Notebook saved to my Drive
- [ ] Capstone step for Day 4 working

<!-- TODO: Mari — recap and preview of next session -->
