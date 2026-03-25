"""Tests for PMJAY FastAPI endpoint validation and behavior."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_navigate_endpoint_success() -> None:
    """Return successful payload for a valid high-confidence request."""

    payload = {
        "query": "What documents are needed for PMJAY?",
        "language": "hi",
        "state": "maharashtra",
        "scheme": "pmjay",
    }
    response = client.post("/api/v1/pmjay/navigate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["fallback"] is False
    assert "notification_id" in body


def test_navigate_endpoint_validation_error_for_short_query() -> None:
    """Reject request when query is shorter than minimum length."""

    payload = {
        "query": "hi",
        "language": "mr",
        "state": "maharashtra",
        "scheme": "pmjay",
    }
    response = client.post("/api/v1/pmjay/navigate", json=payload)

    assert response.status_code == 422


def test_navigate_endpoint_validation_error_for_unsupported_language() -> None:
    """Reject request when language is outside MVP scope."""

    payload = {
        "query": "What is PMJAY eligibility?",
        "language": "en",
        "state": "maharashtra",
        "scheme": "pmjay",
    }
    response = client.post("/api/v1/pmjay/navigate", json=payload)

    assert response.status_code == 422


def test_health_endpoint_works() -> None:
    """Return positive health status."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
