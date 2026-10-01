# Day 3: Agents, tools, memory, and safety

**Thursday, Oct 8, 2026 · 2:00–4:00 PM IST**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mari-muthu-k/agent-to-production/blob/main/labs/day3_agents.ipynb)

<!-- TODO: Mari — one-paragraph overview of the day -->

## Syllabus topics

1. Topic 1 <!-- TODO: Mari -->
2. Topic 2 <!-- TODO: Mari -->
3. Topic 3 <!-- TODO: Mari -->
4. Topic 4 <!-- TODO: Mari -->

## Capstone progress

Turn the RAG pipeline into an agent that chooses tools (search the paper, define a term) and remembers the conversation.

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
    while True:
        action = agent.next_action(history)
        result = TOOLS[action.name](**action.args)
        history.append(result)
    ```

=== "Production"

    ```python
    for step in range(MAX_STEPS):  # (1)!
        action = agent.next_action(history)
        if action.name not in ALLOWED_TOOLS:  # (2)!
            raise ToolNotAllowed(action.name)
        result = TOOLS[action.name](**validate(action.args))
        history.append(result)
        if action.is_final:
            break
    ```

    1. Cap the loop so a confused agent can't run forever.
    2. Allow-list tools and validate their arguments.

!!! warning "Production"
    <!-- TODO: Mari — hardening notes for Day 3 -->
    Hardening notes coming soon.

!!! failure "What goes wrong in production"
    <!-- TODO: Mari — failure story for Day 3 -->
    Failure story coming soon.

## Wrap-up

- [ ] Notebook saved to my Drive
- [ ] Capstone step for Day 3 working

<!-- TODO: Mari — recap and preview of next session -->
