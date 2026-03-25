"""LangChain-oriented agent orchestration for YojanaSetu query answering."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from statistics import mean
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

from api.models import QueryRequest, QueryResponse
from api.retriever import HybridRetriever


class CitationError(ValueError):
    """Raised when a non-fallback answer does not include notification citation."""


@dataclass
class AgentContext:
    """Structured context bundle used to render final prompt."""

    language: str
    question: str
    documents: List[Document]


class YojanaAgent:
    """RAG orchestration class for PMJAY question answering."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_callable: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize agent with retriever and optional LLM callable override."""

        self.retriever = retriever
        self.llm_callable = llm_callable
        self._openai_client = None
        if llm_callable is None and os.getenv("OPENAI_API_KEY", "").strip():
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _classify_intent(self, question: str) -> str:
        """Classify high-level intent for eligibility, scheme, or empanelment queries."""

        lowered = question.lower()
        if any(token in lowered for token in ["eligible", "eligibility", "पात्र", "patra"]):
            return "eligibility"
        if any(token in lowered for token in ["scheme", "pmjay", "yojana", "योजना"]):
            return "scheme"
        if any(token in lowered for token in ["empanelled", "hospital", "अस्पताल", "रुग्णालय"]):
            return "empanelment"
        return "other"

    def _confidence(self, documents: List[Document]) -> float:
        """Compute confidence as mean of top three combined relevance scores."""

        if not documents:
            return 0.0
        scores = sorted(
            [float(doc.metadata.get("combined_score", 0.0)) for doc in documents],
            reverse=True,
        )
        top_scores = scores[:3]
        return max(0.0, min(1.0, float(mean(top_scores))))

    def _render_prompt(self, context: AgentContext) -> str:
        """Create the final LLM prompt using mandated system instruction."""

        system_prompt = (
            "You are YojanaSetu, a helpful healthcare scheme assistant for India. "
            "Answer ONLY from the provided context. You MUST cite the notification_id. "
            "If the answer is not in the context, say so clearly and provide helpline 14555. "
            "Respond in {language}. Be simple, warm, and under 3 sentences."
        )
        context_text = "\n\n".join(
            [
                f"notification_id: {doc.metadata.get('notification_id', 'unknown')}\ncontext: {doc.page_content}"
                for doc in context.documents
            ]
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context_text}\n\n"
                    "Return concise answer with explicit notification_id.",
                ),
            ]
        )
        rendered = prompt_template.format_messages(
            language=context.language,
            question=context.question,
            context_text=context_text,
        )
        return "\n".join([message.content for message in rendered])

    def _call_llm(self, prompt: str) -> str:
        """Invoke LLM either through injected callable or OpenAI fallback client."""

        if self.llm_callable is not None:
            return self.llm_callable(prompt)
        if self._openai_client is None:
            raise RuntimeError("No LLM callable configured and OPENAI_API_KEY is missing.")

        completion = self._openai_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content or ""

    def _extract_notification_id(self, answer: str) -> str:
        """Extract notification id from generated answer and fail if absent."""

        match = re.search(
            r"(?:notification[_\s-]?id\s*[:#-]?\s*)([A-Za-z0-9._/-]+)|\b([A-Z]{2,}-[A-Z]{2,}-\d{4}-\d{3,})\b",
            answer,
            re.IGNORECASE,
        )
        if not match:
            raise CitationError("Generated answer is missing notification_id citation.")
        return (match.group(1) or match.group(2) or "").strip()

    def answer(self, request: QueryRequest) -> QueryResponse:
        """Answer query using intent classification, retrieval, and guarded generation."""

        intent = self._classify_intent(request.question)
        documents = self.retriever.retrieve(
            query=request.question,
            top_k=5,
            hospital_name=request.hospital_name,
        )
        confidence = self._confidence(documents)

        hospital_status = "unknown"
        if request.hospital_name:
            hospital_status = str(self.retriever.hospital_status(request.hospital_name)["hospital_status"])

        if confidence < 0.7 or intent == "other":
            return QueryResponse(
                answer=None,
                notification_id="N/A",
                confidence=confidence,
                hospital_status=hospital_status,
                fallback=True,
                helpline="14555",
                language=request.language,
            )

        prompt = self._render_prompt(
            AgentContext(
                language=request.language,
                question=request.question,
                documents=documents,
            )
        )
        answer_text = self._call_llm(prompt)
        notification_id = self._extract_notification_id(answer_text)

        return QueryResponse(
            answer=answer_text,
            notification_id=notification_id,
            confidence=confidence,
            hospital_status=hospital_status,
            fallback=False,
            helpline=None,
            language=request.language,
        )
