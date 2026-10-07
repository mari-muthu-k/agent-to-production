"""Long-term memory (Day 3, 4.4): notes about each reader in a store, read into the system prompt.

Short-term memory is the checkpointer (one conversation per thread_id). Long-term memory is this store,
keyed by user_id, so a reader's profile follows them into every new conversation. In the notebook the
store lives in memory and is gone when the runtime restarts; Day 4 moves it to a database.
"""
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.store.memory import InMemoryStore

STORE = InMemoryStore()


def read_notes(store, user_id: str) -> list:
    """Every note saved for this reader, oldest first."""
    items = store.search(("readers", user_id), limit=1000)
    return [item.value["note"] for item in sorted(items, key=lambda i: i.created_at)]


@dynamic_prompt
def reader_profile(request: ModelRequest) -> str:
    """Before each model call: the system prompt, plus this reader's saved notes (if any)."""
    base = request.system_prompt or ""
    store, context = request.runtime.store, request.runtime.context
    notes = read_notes(store, context.user_id) if store is not None and context is not None else []
    if not notes:
        return base
    return base + "\n\nReader profile (saved notes; follow it):\n" + "\n".join(f"- {n}" for n in notes)
