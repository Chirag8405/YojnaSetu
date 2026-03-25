"""CSV ingestion utilities for PMJAY hospital empanelment records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_COLUMNS = {
    "hospital_name",
    "city",
    "district",
    "pmjay_id",
    "specialisations",
    "empanelled_date",
}


@dataclass
class DocumentChunk:
    """Structured chunk for vector upsert with required metadata."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def _parse_csv_date(raw_value: str) -> str:
    """Normalize date-like input to ISO format for consistent metadata."""

    text_value = raw_value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text_value, fmt).date().isoformat()
        except ValueError:
            continue
    return text_value


def load_csv_chunks(csv_path: str) -> List[DocumentChunk]:
    """Load PMJAY empanelment CSV rows into structured text chunks."""

    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Invalid CSV path: {csv_path}")

    chunks: List[DocumentChunk] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV has no headers.")

        headers = {header.strip() for header in reader.fieldnames}
        missing = EXPECTED_COLUMNS - headers
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for idx, row in enumerate(reader):
            hospital_name = row["hospital_name"].strip()
            city = row["city"].strip()
            pmjay_id = row["pmjay_id"].strip()
            specialisations = row["specialisations"].strip()
            empanelled_date = row["empanelled_date"].strip()
            normalized_date = _parse_csv_date(empanelled_date)

            if not pmjay_id:
                raise ValueError("pmjay_id is required for every CSV row.")

            text = (
                f"Hospital: {hospital_name} in {city} is empanelled under PMJAY "
                f"(ID: {pmjay_id}) for: {specialisations}. "
                f"Last verified: {empanelled_date}."
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=f"csv:{pmjay_id}:{idx}",
                    text=text,
                    metadata={
                        "source_doc": path.name,
                        "notification_id": pmjay_id,
                        "state": "maharashtra",
                        "last_updated": normalized_date,
                    },
                )
            )

    return chunks
