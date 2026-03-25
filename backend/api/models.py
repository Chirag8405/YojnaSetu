"""Pydantic models for YojanaSetu Phase 2 query APIs."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming request payload for PMJAY RAG questions."""

    question: str = Field(min_length=3, max_length=500)
    language: Literal["hi", "mr"]
    hospital_name: Optional[str] = None


class QueryResponse(BaseModel):
    """Structured answer payload returned by the RAG agent."""

    answer: Optional[str]
    notification_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    hospital_status: Literal["empanelled", "not_empanelled", "unknown"]
    fallback: bool
    helpline: Optional[str]
    language: str


class DenialRequest(BaseModel):
    """Input payload for denial decoder analysis."""

    text: str = Field(min_length=10, max_length=4000)


class DenialResponse(BaseModel):
    """Structured denial decoder result for user guidance."""

    explanation: str
    grievance_number: str = "14555"
    language: Literal["hi", "mr"]
