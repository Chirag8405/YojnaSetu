"""RAG service with strict confidence and citation guardrails."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from app.core.config import settings
from app.models.navigation import (
    FallbackResponse,
    NavigationRequest,
    NavigationResponse,
    SuccessResponse,
)
from app.services.i18n import is_supported_i18n_key


class CitationMissingError(ValueError):
    """Raised when a non-fallback answer does not include a citation."""


@dataclass
class RetrievalResult:
    """Normalized retrieval output from vector DB plus LLM post-processing."""

    answer_i18n_key: str
    answer_params: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    notification_id: Optional[str] = None


def mock_pmjay_retriever(request: NavigationRequest) -> RetrievalResult:
    """Return deterministic PMJAY retrieval output for MVP until live RAG is wired."""

    lowered_query = request.query.lower()
    if "document" in lowered_query or "दस्त" in lowered_query:
        return RetrievalResult(
            answer_i18n_key="pmjay.documents.required",
            answer_params={"state": request.state.value},
            confidence=0.88,
            notification_id="PMJAY-MH-2024-001",
        )
    if "eligible" in lowered_query or "पात्र" in lowered_query:
        return RetrievalResult(
            answer_i18n_key="pmjay.eligibility.summary",
            answer_params={"scheme": request.scheme.value},
            confidence=0.91,
            notification_id="PMJAY-MH-2024-002",
        )
    return RetrievalResult(
        answer_i18n_key="pmjay.eligibility.summary",
        answer_params={},
        confidence=0.52,
        notification_id="PMJAY-MH-2024-003",
    )


def generate_navigation_response(
    request: NavigationRequest,
    retriever: Optional[Callable[[NavigationRequest], RetrievalResult]] = None,
) -> NavigationResponse:
    """Generate a PMJAY response with mandatory citation and confidence fallback."""

    active_retriever = retriever or mock_pmjay_retriever
    result = active_retriever(request)

    if result.confidence < settings.confidence_threshold:
        return FallbackResponse()

    if not result.notification_id:
        raise CitationMissingError("Missing notification_id for non-fallback response.")

    if not is_supported_i18n_key(result.answer_i18n_key):
        raise ValueError(f"Unsupported i18n key returned by retriever: {result.answer_i18n_key}")

    return SuccessResponse(
        answer_i18n_key=result.answer_i18n_key,
        answer_params=result.answer_params,
        fallback=False,
        notification_id=result.notification_id,
        confidence=result.confidence,
    )
