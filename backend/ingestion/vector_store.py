"""Pinecone vector store integration for PMJAY ingestion chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


@dataclass
class VectorRecord:
    """Record shape expected by vector store upsert method."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class VectorStoreClient:
    """Thin client for Pinecone index management, embeddings, and staleness checks."""

    def __init__(self, index_name: str = "yojanasetu-pmjay", namespace: str = "default") -> None:
        """Initialize Pinecone and OpenAI clients from environment variables."""

        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required.")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        self.index_name = index_name
        self.namespace = namespace
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.index = self._ensure_index(index_name=index_name)

    def _ensure_index(self, index_name: str) -> Any:
        """Create index when missing and return index handle."""

        existing_indexes = self.pc.list_indexes().names()
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return self.pc.Index(index_name)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with OpenAI text-embedding-3-small model."""

        response = self.openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]

    def upsert_chunks(self, chunks: Iterable[VectorRecord]) -> int:
        """Upsert chunk records with embeddings into Pinecone index."""

        records = list(chunks)
        if not records:
            return 0

        batch_size = 100
        upserted = 0
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            for record in batch:
                notification_id = str(record.metadata.get("notification_id", "")).strip()
                if not notification_id:
                    raise ValueError(f"Missing notification_id for chunk {record.chunk_id}")
            embeddings = self._embed_texts([record.text for record in batch])
            vectors = [
                {
                    "id": record.chunk_id,
                    "values": embedding,
                    "metadata": {**record.metadata, "text": record.text},
                }
                for record, embedding in zip(batch, embeddings)
            ]
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            upserted += len(vectors)

        return upserted

    def _iter_all_ids(self) -> List[str]:
        """List all vector ids from namespace for fetch-based metadata scan."""

        vector_ids: List[str] = []
        listed = self.index.list(namespace=self.namespace)
        for page in listed:
            ids = getattr(page, "vectors", None)
            if ids is None and isinstance(page, dict):
                ids = page.get("vectors", [])
            for item in ids or []:
                if isinstance(item, str):
                    vector_ids.append(item)
                elif isinstance(item, dict) and "id" in item:
                    vector_ids.append(item["id"])
        return vector_ids

    def _fetch_all_metadata(self) -> List[Dict[str, Any]]:
        """Fetch metadata for all vectors currently in namespace."""

        ids = self._iter_all_ids()
        if not ids:
            return []

        response = self.index.fetch(ids=ids, namespace=self.namespace)
        vectors = response.get("vectors", {}) if isinstance(response, dict) else getattr(response, "vectors", {})

        metadata_rows: List[Dict[str, Any]] = []
        for vector in vectors.values():
            metadata = vector.get("metadata", {}) if isinstance(vector, dict) else getattr(vector, "metadata", {})
            if metadata:
                metadata_rows.append(metadata)
        return metadata_rows

    def fetch_all_records(self) -> List[Dict[str, Any]]:
        """Fetch all records with text and metadata for lexical retrieval."""

        ids = self._iter_all_ids()
        if not ids:
            return []

        response = self.index.fetch(ids=ids, namespace=self.namespace)
        vectors = response.get("vectors", {}) if isinstance(response, dict) else getattr(response, "vectors", {})

        records: List[Dict[str, Any]] = []
        for vector_id, vector in vectors.items():
            metadata = vector.get("metadata", {}) if isinstance(vector, dict) else getattr(vector, "metadata", {})
            text = str(metadata.get("text", "")).strip()
            records.append(
                {
                    "id": vector_id,
                    "text": text,
                    "metadata": metadata,
                }
            )
        return records

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run semantic query in Pinecone and return scored document records."""

        query_vector = self._embed_texts([query])[0]
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=metadata_filter,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])

        documents: List[Dict[str, Any]] = []
        for match in matches:
            match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", "")
            score = match.get("score") if isinstance(match, dict) else getattr(match, "score", 0.0)
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            documents.append(
                {
                    "id": str(match_id),
                    "score": float(score or 0.0),
                    "text": str(metadata.get("text", "")),
                    "metadata": metadata,
                }
            )
        return documents

    def count_records(self) -> int:
        """Return total vector count for this namespace."""

        stats = self.index.describe_index_stats()
        namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})
        namespace_data = namespaces.get(self.namespace, {}) if isinstance(namespaces, dict) else {}
        if isinstance(namespace_data, dict):
            return int(namespace_data.get("vector_count", 0))
        return int(getattr(namespace_data, "vector_count", 0))

    def staleness_check(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Return staleness flags where records older than 7 days are marked stale."""

        current_time = now or datetime.now(timezone.utc)
        cutoff = current_time.date() - timedelta(days=7)
        results: List[Dict[str, Any]] = []

        for metadata in self._fetch_all_metadata():
            raw_last_updated = str(metadata.get("last_updated", "")).strip()
            try:
                parsed_date = date.fromisoformat(raw_last_updated)
            except ValueError:
                parsed_date = current_time.date()

            results.append(
                {
                    "source_doc": metadata.get("source_doc", "unknown"),
                    "last_updated": raw_last_updated,
                    "is_stale": parsed_date < cutoff,
                }
            )

        return results
