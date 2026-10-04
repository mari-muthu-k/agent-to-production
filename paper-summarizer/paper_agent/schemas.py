"""Output contracts for the Paper Summarizer (Day 1, section 5.2)."""
from typing import Literal

from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="Plain-language answer, 1-3 sentences")
    citations: list[str] = Field(min_length=1, description="Section ids used, e.g. ['results']")
    confidence: Literal["high", "medium", "low"]


class KeyTerm(BaseModel):
    term: str
    plain_definition: str = Field(description="One sentence a first-year CS student understands")


class PaperExplainer(BaseModel):
    headline: str = Field(description="Plain-language headline, max 12 words, no hype")
    summary: str = Field(description="3-5 short sentences in plain language")
    why_it_matters: str
    evidence_type: Literal["benchmark", "user_study", "simulation", "theoretical", "deployment", "survey"]
    key_terms: list[KeyTerm] = Field(min_length=1, max_length=5)
    caveats: list[str] = Field(min_length=1, description="Limitations the reader must know")
    citations: list[str] = Field(min_length=1, description="Section ids used")
