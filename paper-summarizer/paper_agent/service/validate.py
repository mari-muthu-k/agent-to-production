"""Upload checks (Day 4, 4.3: TODO 1 in the notebook). The filename and the Content-Type header both come
from the client, so neither proves anything; the first bytes of the file do. Size is checked before
parsing: a huge or broken file is exactly the work we want to avoid."""
from fastapi import HTTPException

from paper_agent.service.config import MAX_UPLOAD_MB


def validate_upload(data: bytes) -> None:
    """415 unless the bytes start like a PDF; 413 if they are over MAX_UPLOAD_MB."""
    if not data.startswith(b"%PDF-"): raise HTTPException(415, "Only PDF files are accepted.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024: raise HTTPException(413, f"Max upload size is {MAX_UPLOAD_MB} MB.")
