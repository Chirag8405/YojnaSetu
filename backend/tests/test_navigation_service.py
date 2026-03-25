"""Tests for PMJAY RAG service behavior."""

import pytest

from app.models.navigation import NavigationRequest
from app.services.rag_service import (
    CitationMissingError,
    RetrievalResult,
    generate_navigation_response,
)


def test_service_returns_success_with_notification_id() -> None:
    """Return successful response when confidence is high and citation exists."""

    request = NavigationRequest(query="Am I eligible for PMJAY?", language="hi")
    response = generate_navigation_response(request)

    assert response.fallback is False
    assert response.notification_id.startswith("PMJAY-MH")


def test_service_returns_fallback_when_confidence_low() -> None:
    """Return fallback payload when confidence falls below threshold."""

    request = NavigationRequest(query="Unclear request about unknown policy", language="mr")
    response = generate_navigation_response(request)

    assert response.fallback is True
    assert response.helpline == "14555"


def test_service_returns_fallback_for_unknown_question() -> None:
    """Return fallback payload for unknown or vague question."""

    request = NavigationRequest(query="Tell me random scheme details", language="mr")
    response = generate_navigation_response(request)

    assert response.fallback is True
    assert response.helpline == "14555"


def test_service_raises_when_citation_missing() -> None:
    """Raise error if non-fallback answer does not include notification_id."""

    def missing_citation_retriever(_: NavigationRequest) -> RetrievalResult:
        """Return high-confidence answer without citation for negative test."""

        return RetrievalResult(
            answer_i18n_key="pmjay.eligibility.summary",
            confidence=0.95,
            notification_id=None,
        )

    request = NavigationRequest(query="eligible", language="hi")
    with pytest.raises(CitationMissingError):
        generate_navigation_response(request, retriever=missing_citation_retriever)
