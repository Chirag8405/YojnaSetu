"""Hybrid retrieval layer combining Pinecone semantic search and BM25 keyword search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ingestion.vector_store import VectorStoreClient


@dataclass
class RetrievalRecord:
    """Intermediate retrieval record used for score fusion."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    bm25_score: float = 0.0


class HybridRetriever:
    """Hybrid retriever with semantic and lexical ranking plus score fusion."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreClient] = None,
        local_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize retriever from either vector store records or local records."""

        self.vector_store = vector_store
        self.local_documents = local_documents or []
        self._bm25 = None
        self._bm25_records: List[Dict[str, Any]] = []

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load candidate records for lexical retrieval."""

        if self.local_documents:
            return self.local_documents
        if self.vector_store is None:
            return []
        return self.vector_store.fetch_all_records()

    def _build_bm25(self) -> None:
        """Build BM25 index over available records if not already built."""

        self._bm25_records = self._load_records()
        tokenized = [record.get("text", "").lower().split() for record in self._bm25_records]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _semantic_results(self, query: str, top_k: int) -> List[RetrievalRecord]:
        """Run Pinecone semantic retrieval and convert matches to internal records."""

        if self.vector_store is None:
            return []

        records: List[RetrievalRecord] = []
        for item in self.vector_store.semantic_search(query=query, top_k=top_k):
            records.append(
                RetrievalRecord(
                    chunk_id=item["id"],
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}),
                    semantic_score=float(item.get("score", 0.0)),
                )
            )
        return records

    def _bm25_results(self, query: str, top_k: int) -> List[RetrievalRecord]:
        """Run BM25 lexical retrieval over locally available corpus."""

        if self._bm25 is None:
            self._build_bm25()
        if self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

        max_score = max([score for _, score in ranked], default=1.0) or 1.0
        results: List[RetrievalRecord] = []
        for index, score in ranked:
            record = self._bm25_records[index]
            normalized = float(score) / float(max_score)
            results.append(
                RetrievalRecord(
                    chunk_id=str(record.get("id", f"record-{index}")),
                    text=str(record.get("text", "")),
                    metadata=dict(record.get("metadata", {})),
                    bm25_score=normalized,
                )
            )
        return results

    def retrieve(self, query: str, top_k: int = 5, hospital_name: Optional[str] = None) -> List[Document]:
        """Retrieve top documents with fused semantic and BM25 relevance scoring."""

        merged: Dict[str, RetrievalRecord] = {}

        for item in self._semantic_results(query=query, top_k=top_k):
            merged[item.chunk_id] = item

        for item in self._bm25_results(query=query, top_k=top_k):
            if item.chunk_id in merged:
                merged[item.chunk_id].bm25_score = item.bm25_score
            else:
                merged[item.chunk_id] = item

        documents: List[Document] = []
        hospital_query = (hospital_name or "").strip().lower()
        for record in merged.values():
            combined = (0.6 * record.semantic_score) + (0.4 * record.bm25_score)
            metadata = {
                **record.metadata,
                "chunk_id": record.chunk_id,
                "semantic_score": record.semantic_score,
                "bm25_score": record.bm25_score,
                "combined_score": combined,
                "hospital_priority": bool(hospital_query and hospital_query in record.text.lower()),
            }
            documents.append(Document(page_content=record.text, metadata=metadata))

        documents.sort(
            key=lambda doc: (
                not bool(doc.metadata.get("hospital_priority", False)),
                -float(doc.metadata.get("combined_score", 0.0)),
            )
        )
        return documents[:top_k]

    def hospital_status(self, hospital_name: str) -> Dict[str, Optional[str]]:
        """Resolve hospital empanelment status from indexed hospital records."""

        candidate_name = hospital_name.strip().lower()
        records = self._load_records()
        matches = [record for record in records if candidate_name and candidate_name in str(record.get("text", "")).lower()]

        if not matches:
            return {"hospital_status": "unknown", "last_verified": None}

        first_match = matches[0]
        raw_text = str(first_match.get("text", "")).lower()
        status = "not_empanelled" if "not empanelled" in raw_text else "empanelled"
        metadata = first_match.get("metadata", {})
        last_verified = str(metadata.get("last_updated", "")).strip() or None
        return {"hospital_status": status, "last_verified": last_verified}
