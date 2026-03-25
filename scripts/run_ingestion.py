"""CLI entrypoint for Phase 1 ingestion into Pinecone vector store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ingestion.csv_loader import load_csv_chunks
from ingestion.pdf_loader import load_pdf_chunks
from ingestion.vector_store import VectorRecord, VectorStoreClient


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for ingestion run command."""

    parser = argparse.ArgumentParser(description="Run YojanaSetu Phase 1 ingestion")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PMJAY PDF files")
    parser.add_argument("--csv", required=True, help="Path to Maharashtra PMJAY hospital CSV")
    return parser


def _to_vector_records(pdf_chunks: list, csv_chunks: list) -> List[VectorRecord]:
    """Convert loader chunks into vector-store-ready records."""

    return [VectorRecord(chunk_id=item.chunk_id, text=item.text, metadata=item.metadata) for item in [*pdf_chunks, *csv_chunks]]


def main() -> None:
    """Run ingestion from PDFs and CSV and upsert records into Pinecone."""

    parser = _build_arg_parser()
    args = parser.parse_args()

    pdf_chunks = load_pdf_chunks(args.pdf_dir)
    csv_chunks = load_csv_chunks(args.csv)

    print(f"{len(pdf_chunks)} chunks from PDFs")
    print(f"{len(csv_chunks)} chunks from CSV")

    client = VectorStoreClient(index_name="yojanasetu-pmjay")
    upserted_count = client.upsert_chunks(_to_vector_records(pdf_chunks, csv_chunks))
    print(f"{upserted_count} vectors upserted into yojanasetu-pmjay")

    staleness = client.staleness_check()
    print(staleness)


if __name__ == "__main__":
    main()
