"""Day 3, Section 6: caching. Four places to cache, and the key is always the hard part.

keys          normalize(), make_key(): everything that changes the answer goes into the key
tool_cache    cached_search(): search results per (paper content, embedding model, k, query)
semantic      SemanticCache: reworded repeats, by embedding similarity, scoped to one paper
prompt_cache  cached_input_tokens(): what the provider's own prefix cache saved, if it says

The exact-match response cache is LangChain's: ChatOpenAI(cache=InMemoryCache()) (6.2). It stores raw model
replies before any guard runs, so use it for fixed steps (summarize section X), not for agent answers;
answers are cached with SemanticCache, which refuses anything a guard rejected.
"""
