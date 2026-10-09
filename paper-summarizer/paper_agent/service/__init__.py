"""Day 4: the Day 3 agent as a service (guide Sections 4-6).

config    names, limits, prices, GATEWAY_MODE (library | proxy), Colab Secrets
gateway   make_router() (LiteLLM as a library), make_chat_model() (library or proxy), embeddings
errors    BudgetExceededError, RateLimitedError, and the proxy's errors mapped onto them
usage     spend per user, budget_guard (before every model call), record_usage, llm/tool spans
store     PaperStore: paper id = hash of the content; search() in a `retrieve` span
pipeline  build_agent(), ask_paper(), explain_events()
validate  validate_upload() (TODO 1)
app       build_app(): routes, auth, rate limit, error handlers, one log line per request, SSE
tracing   setup_tracing(), show_trace()
"""
