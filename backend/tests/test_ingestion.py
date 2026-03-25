"""Tests for Phase 1 ingestion loaders and staleness check behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitz
import pytest

from ingestion.csv_loader import load_csv_chunks
from ingestion.pdf_loader import load_pdf_chunks
from ingestion.vector_store import VectorStoreClient


class _FakeVectorStore(VectorStoreClient):
    """Minimal test double that bypasses external clients for unit testing."""

    def __init__(self, fake_rows: list[dict[str, str]]) -> None:
        """Store fake metadata rows for deterministic staleness checks."""

        self._fake_rows = fake_rows

    def _fetch_all_metadata(self) -> list[dict[str, str]]:
        """Return fake metadata rows instead of querying Pinecone."""

        return self._fake_rows


def _create_pdf(path: Path, text: str) -> None:
    """Create a simple single-page PDF file with provided text content."""

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_pdf_loader_raises_if_notification_id_missing(tmp_path: Path) -> None:
    """Raise ValueError when PDF does not include notification id in header/footer region."""

    pdf_path = tmp_path / "missing_notification.pdf"
    _create_pdf(pdf_path, "PMJAY circular without a valid notification identifier")

    with pytest.raises(ValueError, match="notification_id"):
        load_pdf_chunks(str(tmp_path))


def test_csv_loader_formats_hospital_text_chunks(tmp_path: Path) -> None:
    """Format CSV row into expected PMJAY hospital text chunk string."""

    csv_path = tmp_path / "hospitals.csv"
    csv_path.write_text(
        "hospital_name,city,district,pmjay_id,specialisations,empanelled_date\n"
        "Sahyadri Hospital,Pune,Pune,PMJAY-MH-H001,Cardiology;Oncology,2026-03-01\n",
        encoding="utf-8",
    )

    chunks = load_csv_chunks(str(csv_path))

    assert len(chunks) == 1
    assert chunks[0].text == (
        "Hospital: Sahyadri Hospital in Pune is empanelled under PMJAY "
        "(ID: PMJAY-MH-H001) for: Cardiology;Oncology. Last verified: 2026-03-01."
    )
    assert chunks[0].metadata["notification_id"] == "PMJAY-MH-H001"


def test_staleness_check_identifies_records_older_than_7_days() -> None:
    """Mark records stale when last_updated is more than seven days old."""

    now = datetime(2026, 3, 25, tzinfo=timezone.utc)
    fresh_date = (now.date() - timedelta(days=2)).isoformat()
    stale_date = (now.date() - timedelta(days=10)).isoformat()

    fake_client = _FakeVectorStore(
        [
            {
                "source_doc": "fresh.pdf",
                "last_updated": fresh_date,
            },
            {
                "source_doc": "stale.pdf",
                "last_updated": stale_date,
            },
        ]
    )

    result = fake_client.staleness_check(now=now)

    assert isinstance(result, list)
    assert result[0]["is_stale"] is False
    assert result[1]["is_stale"] is True
