"""Day 2 ingestion: load PDFs -> clean + find sections -> split into chunks -> embed -> store.

Submodules are imported directly (e.g. `from paper_agent.ingest.loaders import load_pdf`) so that
light modules such as `hashing` work without the optional `rag` dependencies installed.
"""
