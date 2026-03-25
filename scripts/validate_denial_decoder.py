"""Denial decoder validation for language and grievance compliance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.main import app


def _expected_language(path: Path) -> str:
    """Infer expected language from denial file name suffix."""

    name = path.stem.lower()
    if name.endswith("_mr"):
        return "mr"
    return "hi"


def run_validation(letters_dir: Path) -> Dict[str, Any]:
    """Validate denial endpoint responses for language and grievance number."""

    client = TestClient(app)
    files = sorted([path for path in letters_dir.glob("*.txt") if path.is_file()])

    checked = 0
    language_failures = 0
    grievance_failures = 0

    for file_path in files:
        payload = {"text": file_path.read_text(encoding="utf-8")}
        response = client.post("/denial", json=payload)
        response.raise_for_status()
        body: Dict[str, Any] = response.json()

        expected_language = _expected_language(file_path)
        if body.get("language") != expected_language:
            language_failures += 1

        grievance = str(body.get("grievance_number", ""))
        if "14555" not in grievance:
            grievance_failures += 1

        checked += 1

    return {
        "checked": checked,
        "language_failures": language_failures,
        "grievance_failures": grievance_failures,
        "pass": checked > 0 and language_failures == 0 and grievance_failures == 0,
    }


def main() -> None:
    """CLI entrypoint for denial decoder compliance validation."""

    parser = argparse.ArgumentParser(description="Validate denial decoder outputs.")
    parser.add_argument(
        "--letters-dir",
        default=str(PROJECT_ROOT / "backend" / "tests" / "denial_letters"),
        help="Path to folder with denial letter templates",
    )
    args = parser.parse_args()

    report = run_validation(Path(args.letters_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
