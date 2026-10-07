"""Day 3: the paper summarizer as a LangChain agent (guide Sections 3-6).

papers   Context (user_id, paper_id) and the papers the tools can read, built with Day 2's pipeline
tools    search_paper, get_section, define_term, save_note (+ flaky_search for 4.5)
prompts  agent_system_v1
model    make_chat_model(): ChatOpenAI from the environment, one retry layer
memory   the long-term store and the reader-profile middleware
guards   the five guard layers and the custom middleware
trace    stream_trace() and trace_table(): steps, tokens and cost
build    build_agent()
policy   how the offline models pick a tool (shared with mock-llm)
fakes    an offline chat model for the TODO tests

The notebook shows these functions with src(), in one namespace, so modules use the same names the
notebook does: `day2` is Day 2's pdf_rag module, and every helper is imported by its own name.
Importing this package does not import LangChain (mock-llm imports `policy`).
"""
