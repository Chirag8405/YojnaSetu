"""FastAPI entrypoint for Phase 2 YojanaSetu RAG backend."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.agent import YojanaAgent
from api.models import DenialRequest, DenialResponse, QueryRequest, QueryResponse
from api.retriever import HybridRetriever
from ingestion.vector_store import VectorStoreClient
from voice.call_flow import set_agent as set_voice_agent
from voice.ivr_handler import router as ivr_router

logger = logging.getLogger("yojanasetu.api")
logging.basicConfig(level=logging.INFO)


def _build_agent() -> tuple[YojanaAgent, VectorStoreClient | None, HybridRetriever]:
    """Build agent with vector store when credentials are configured."""

    try:
        vector_store = VectorStoreClient(index_name="yojanasetu-pmjay")
    except ValueError:
        vector_store = None
    retriever = HybridRetriever(vector_store=vector_store, local_documents=[])
    return YojanaAgent(retriever=retriever), vector_store, retriever


agent, vector_store_client, hybrid_retriever = _build_agent()
set_voice_agent(agent)
app = FastAPI(title="YojanaSetu API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ivr_router)


def _detect_denial_language(text: str) -> str:
    """Detect Hindi or Marathi script hints for denial response language."""

    marathi_markers = ["मध्ये", "रुग्ण", "नाकार", "नकार", "आहे", "साठी", "कृपया", "झाला"]
    if any(marker in text for marker in marathi_markers):
        return "mr"
    return "hi"


def _denial_explanation(text: str, language: str) -> str:
    """Create concise denial reason guidance in detected language."""

    lowered = text.lower()
    if language == "mr":
        if "document" in lowered or "कागद" in lowered:
            return "कागदपत्रे अपुरी असल्याने दावा नाकारला जाऊ शकतो. आयुष्मान कार्ड, ओळखपत्र आणि डिस्चार्ज कागदपत्रे घेऊन पुन्हा अर्ज करा."
        if "not empanelled" in lowered or "empanel" in lowered:
            return "रुग्णालय PMJAY अंतर्गत एम्पॅनल नसल्यामुळे नकार मिळाला असू शकतो. जवळच्या एम्पॅनल रुग्णालयाची पडताळणी करा."
        return "नकाराचे कारण स्पष्ट नाही. रुग्णालयाकडून लेखी कारण घ्या आणि PMJAY तक्रार सहाय्यास संपर्क करा."

    if "document" in lowered or "दस्त" in lowered:
        return "दस्तावेज अधूरे होने से दावा अस्वीकार हुआ हो सकता है। आयुष्मान कार्ड, पहचान पत्र और डिस्चार्ज पेपर लेकर दोबारा आवेदन करें।"
    if "not empanelled" in lowered or "empanel" in lowered:
        return "अस्पताल PMJAY में एम्पैनल न होने के कारण अस्वीकार हुआ हो सकता है। नजदीकी एम्पैनल अस्पताल की जांच करें।"
    return "अस्वीकृति का कारण स्पष्ट नहीं है। अस्पताल से लिखित कारण लें और PMJAY शिकायत सहायता से संपर्क करें।"


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any) -> Any:
    """Log session id, question hash, latency, and confidence for each request."""

    start = time.perf_counter()
    session_id = request.headers.get("x-session-id", str(uuid.uuid4()))
    body_bytes = await request.body()
    question_hash = None

    if request.url.path == "/query":
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
            question = str(payload.get("question", ""))
            question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        except Exception:
            question_hash = None

    async def receive() -> Dict[str, Any]:
        """Provide request body back to downstream handler after prior read."""

        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request._receive = receive
    request.state.session_id = session_id
    request.state.question_hash = question_hash

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    confidence = getattr(request.state, "confidence", None)

    logger.info(
        "session_id=%s question_hash=%s response_time_ms=%.2f confidence=%s",
        session_id,
        question_hash,
        elapsed_ms,
        confidence,
    )
    return response


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, raw_request: Request) -> QueryResponse:
    """Answer PMJAY query request with retriever-grounded citation response."""

    response = agent.answer(request)
    raw_request.state.confidence = response.confidence
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    """Return API health, vector record count, and stale source summary."""

    if vector_store_client is None:
        return {"status": "ok", "vector_store_records": 0, "stale_sources": []}

    return {
        "status": "ok",
        "vector_store_records": vector_store_client.count_records(),
        "stale_sources": vector_store_client.staleness_check(),
    }


@app.get("/hospital/{name}")
def hospital_status(name: str) -> Dict[str, Any]:
    """Return empanelment status and last verified date for hospital name."""

    status_payload = hybrid_retriever.hospital_status(name)
    return {
        "hospital_name": name,
        "hospital_status": status_payload["hospital_status"],
        "last_verified": status_payload["last_verified"],
    }


@app.post("/denial", response_model=DenialResponse)
def denial_decoder(request: DenialRequest) -> DenialResponse:
    """Decode rejection text into plain-language guidance and grievance number."""

    language = _detect_denial_language(request.text)
    explanation = _denial_explanation(request.text, language)
    return DenialResponse(explanation=explanation, grievance_number="14555", language=language)
