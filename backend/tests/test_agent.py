"""Tests for YojanaSetu Phase 2 agent orchestration and guardrails."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from fastapi.testclient import TestClient

from api.agent import YojanaAgent
from api.models import QueryRequest, QueryResponse


class _FakeRetriever:
    """Retriever test double with deterministic retrieval and hospital status."""

    def __init__(self, documents: List[Document], hospital_map: Optional[Dict[str, str]] = None) -> None:
        """Store fixed retrieval docs and hospital status mapping."""

        self._documents = documents
        self._hospital_map = hospital_map or {}

    def retrieve(self, query: str, top_k: int = 5, hospital_name: Optional[str] = None) -> List[Document]:
        """Return prepared documents regardless of query for deterministic unit tests."""

        _ = (query, top_k, hospital_name)
        return self._documents[:top_k]

    def hospital_status(self, hospital_name: str) -> Dict[str, Optional[str]]:
        """Resolve predefined hospital status for tests."""

        value = self._hospital_map.get(hospital_name.lower(), "unknown")
        return {"hospital_status": value, "last_verified": "2026-03-01" if value != "unknown" else None}


class _CountingLLM:
    """LLM stub that counts invocations for fallback assertions."""

    def __init__(self, response_text: str) -> None:
        """Initialize with fixed response text for all prompts."""

        self.response_text = response_text
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        """Return configured response and count each invocation."""

        _ = prompt
        self.calls += 1
        return self.response_text


def test_empanelled_hospital_and_condition_returns_answer_with_notification() -> None:
    """Return grounded answer with citation for supported empanelled scenario."""

    docs = [
        Document(
            page_content="Hospital: Ruby Hall in Pune is empanelled under PMJAY.",
            metadata={"combined_score": 0.92, "notification_id": "PMJAY-MH-2024-101"},
        ),
        Document(
            page_content="PMJAY covers listed cardiology procedures.",
            metadata={"combined_score": 0.88, "notification_id": "PMJAY-MH-2024-101"},
        ),
        Document(
            page_content="Eligibility details for PMJAY in Maharashtra.",
            metadata={"combined_score": 0.84, "notification_id": "PMJAY-MH-2024-101"},
        ),
    ]
    llm = _CountingLLM("Hospital is covered. notification_id: PMJAY-MH-2024-101")
    retriever = _FakeRetriever(docs, hospital_map={"ruby hall": "empanelled"})
    agent = YojanaAgent(retriever=retriever, llm_callable=llm)

    response = agent.answer(
        QueryRequest(question="Is Ruby Hall covered for cardiology under PMJAY?", language="hi", hospital_name="Ruby Hall")
    )

    assert response.fallback is False
    assert response.answer is not None
    assert response.notification_id == "PMJAY-MH-2024-101"
    assert response.hospital_status == "empanelled"
    assert llm.calls == 1


def test_unknown_hospital_returns_unknown_status_without_false_claim() -> None:
    """Mark hospital status unknown when hospital record is not found."""

    docs = [
        Document(
            page_content="No direct hospital record in retrieved context.",
            metadata={"combined_score": 0.85, "notification_id": "PMJAY-MH-2024-102"},
        ),
        Document(
            page_content="General PMJAY context for Maharashtra only.",
            metadata={"combined_score": 0.81, "notification_id": "PMJAY-MH-2024-102"},
        ),
        Document(
            page_content="Scheme guidance and process.",
            metadata={"combined_score": 0.79, "notification_id": "PMJAY-MH-2024-102"},
        ),
    ]
    llm = _CountingLLM("I cannot verify this hospital from the context. notification_id: PMJAY-MH-2024-102")
    retriever = _FakeRetriever(docs, hospital_map={})
    agent = YojanaAgent(retriever=retriever, llm_callable=llm)

    response = agent.answer(
        QueryRequest(question="Is UnknownCare Hospital empanelled?", language="mr", hospital_name="UnknownCare Hospital")
    )

    assert response.hospital_status == "unknown"
    assert response.notification_id == "PMJAY-MH-2024-102"
    assert "cannot verify" in (response.answer or "").lower()


def test_ambiguous_query_triggers_fallback_and_skips_llm() -> None:
    """Skip LLM calls when confidence is below threshold and return fallback."""

    docs = [
        Document(page_content="Weakly related text", metadata={"combined_score": 0.40, "notification_id": "PMJAY-MH-2024-201"}),
        Document(page_content="Another weak chunk", metadata={"combined_score": 0.35, "notification_id": "PMJAY-MH-2024-202"}),
        Document(page_content="Low confidence snippet", metadata={"combined_score": 0.20, "notification_id": "PMJAY-MH-2024-203"}),
    ]
    llm = _CountingLLM("This should never be called")
    retriever = _FakeRetriever(docs)
    agent = YojanaAgent(retriever=retriever, llm_callable=llm)

    response = agent.answer(QueryRequest(question="Tell me something", language="hi", hospital_name=None))

    assert response.fallback is True
    assert response.answer is None
    assert response.confidence < 0.7
    assert llm.calls == 0


def test_non_pmjay_question_triggers_fallback() -> None:
    """Return fallback for non-PMJAY intent without unsupported claims."""

    docs = [
        Document(page_content="Unrelated context", metadata={"combined_score": 0.95, "notification_id": "PMJAY-MH-2024-301"}),
        Document(page_content="Another unrelated context", metadata={"combined_score": 0.90, "notification_id": "PMJAY-MH-2024-301"}),
        Document(page_content="More unrelated context", metadata={"combined_score": 0.88, "notification_id": "PMJAY-MH-2024-301"}),
    ]
    llm = _CountingLLM("Should not be called for non PMJAY question")
    retriever = _FakeRetriever(docs)
    agent = YojanaAgent(retriever=retriever, llm_callable=llm)

    response = agent.answer(QueryRequest(question="Who won the football match yesterday?", language="hi", hospital_name=None))

    assert response.fallback is True
    assert response.helpline == "14555"
    assert llm.calls == 0


def test_post_query_non_fallback_response_contains_notification_id() -> None:
    """Ensure /query always includes notification_id for non-fallback responses."""

    from api import main as api_main

    class _FakeAgent:
        """Minimal agent stub to force deterministic /query response."""

        def answer(self, request: QueryRequest) -> Any:
            """Return deterministic non-fallback payload for API assertion."""

            _ = request
            return QueryResponse(
                answer="PMJAY details are available. notification_id: PMJAY-MH-2024-999",
                notification_id="PMJAY-MH-2024-999",
                confidence=0.91,
                hospital_status="unknown",
                fallback=False,
                helpline=None,
                language="hi",
            )

    previous_agent = api_main.agent
    api_main.agent = _FakeAgent()
    try:
        client = TestClient(api_main.app)
        response = client.post(
            "/query",
            json={"question": "PMJAY eligibility?", "language": "hi", "hospital_name": None},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["fallback"] is False
        assert bool(body["notification_id"]) is True
    finally:
        api_main.agent = previous_agent


def test_get_health_returns_valid_json() -> None:
    """Ensure /health endpoint returns expected JSON structure."""

    from api import main as api_main

    class _FakeVectorStore:
        """Fake vector store to provide deterministic health values."""

        def count_records(self) -> int:
            """Return fixed record count for testing."""

            return 3

        def staleness_check(self) -> List[Dict[str, Any]]:
            """Return fixed stale-source list for testing."""

            return [{"source_doc": "sample.pdf", "last_updated": "2026-03-20", "is_stale": False}]

    previous_vector_store = api_main.vector_store_client
    api_main.vector_store_client = _FakeVectorStore()
    try:
        client = TestClient(api_main.app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["vector_store_records"], int)
        assert isinstance(body["stale_sources"], list)
    finally:
        api_main.vector_store_client = previous_vector_store
