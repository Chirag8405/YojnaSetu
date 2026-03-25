"""PDF ingestion utilities for PMJAY Maharashtra notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List

import fitz
import tiktoken


NOTIFICATION_LABEL_PATTERN = re.compile(
    r"(?:notification\s*(?:id|no|number)\b\s*[:#-]?\s*)([A-Za-z0-9._/-]+)",
    re.IGNORECASE,
)
NOTIFICATION_CODE_PATTERN = re.compile(r"\b[A-Z]{2,}-[A-Z]{2,}-\d{4}-\d{3,}\b")


@dataclass
class DocumentChunk:
    """Structured chunk for vector upsert with required metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def _chunk_token_ids(token_ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    """Split token ids into overlapping windows with deterministic boundaries."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    windows: List[List[int]] = []
    step = max_tokens - overlap
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + max_tokens]
        if not window:
            continue
        windows.append(window)
        if start + max_tokens >= len(token_ids):
            break
    return windows


def _extract_notification_id(raw_text: str) -> str:
    """Extract notification id from header/footer-like regions of page text."""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("PDF has no readable lines for notification_id extraction.")

    candidate_region = lines[:8] + lines[-8:]
    region_text = "\n".join(candidate_region)

    label_match = NOTIFICATION_LABEL_PATTERN.search(region_text)
    if label_match:
        return label_match.group(1)

    code_match = NOTIFICATION_CODE_PATTERN.search(region_text)
    if code_match:
        return code_match.group(0)

    raise ValueError("notification_id could not be found in document header/footer.")


def _read_pdf_text(pdf_path: Path) -> str:
    """Read and concatenate all page text from a PDF file."""

    contents: List[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            contents.append(page.get_text("text"))
    return "\n".join(contents).strip()


def load_pdf_chunks(pdf_dir: str, max_tokens: int = 500, overlap: int = 50) -> List[DocumentChunk]:
    """Load PDFs from folder and split each document into token-overlap chunks."""

    base_dir = Path(pdf_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        raise ValueError(f"Invalid PDF directory: {pdf_dir}")

    encoder = tiktoken.get_encoding("cl100k_base")
    chunks: List[DocumentChunk] = []

    for pdf_path in sorted(base_dir.glob("*.pdf")):
        raw_text = _read_pdf_text(pdf_path)
        if not raw_text:
            continue

        notification_id = _extract_notification_id(raw_text)
        token_ids = encoder.encode(raw_text)
        windows = _chunk_token_ids(token_ids, max_tokens=max_tokens, overlap=overlap)
        file_last_updated = datetime.fromtimestamp(pdf_path.stat().st_mtime, tz=timezone.utc).date()

        for idx, token_window in enumerate(windows):
            chunk_text = encoder.decode(token_window).strip()
            if not chunk_text:
                continue
            chunk = DocumentChunk(
                chunk_id=f"pdf:{pdf_path.stem}:{idx}",
                text=chunk_text,
                metadata={
                    "source_doc": pdf_path.name,
                    "notification_id": notification_id,
                    "state": "maharashtra",
                    "last_updated": file_last_updated.isoformat(),
                },
            )
            chunks.append(chunk)

    return chunks
