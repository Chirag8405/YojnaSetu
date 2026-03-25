"""Golden set validation runner for YojanaSetu Phase 5."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import sqrt
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.main import agent
from api.models import QueryRequest


def _tokenize(text: str) -> List[str]:
    """Convert text into normalized token list for cosine matching."""

    return [token.strip() for token in text.lower().replace("\n", " ").split() if token.strip()]


def _cosine_similarity(left: str, right: str) -> float:
    """Compute bag-of-words cosine similarity between two text snippets."""

    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    vocabulary = set(left_counts.keys()).union(right_counts.keys())

    numerator = sum(left_counts[token] * right_counts[token] for token in vocabulary)
    left_norm = sqrt(sum(value * value for value in left_counts.values()))
    right_norm = sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _answer_matches(actual_answer: str, expected_benefit: str) -> bool:
    """Check exact or semantic match against expected benefit text."""

    if not expected_benefit.strip():
        return False
    actual_normalized = actual_answer.strip().lower()
    expected_normalized = expected_benefit.strip().lower()
    if expected_normalized in actual_normalized:
        return True
    return _cosine_similarity(actual_answer, expected_benefit) >= 0.72


def run_validation(golden_set_path: Path) -> Dict[str, Any]:
    """Execute golden set validation and compute quality metrics."""

    payload = json.loads(golden_set_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("golden_set.json must contain a list of test records.")

    tp = fp = tn = fn = fallback_count = citation_missing = 0

    for row in payload:
        if not isinstance(row, dict):
            continue
        query = str(row.get("query", "")).strip()
        expected_benefit = str(row.get("expected_benefit", "")).strip()
        language = str(row.get("language", "hi")).strip()
        if language not in {"hi", "mr"}:
            language = "hi"
        if not query:
            continue

        response = agent.answer(QueryRequest(question=query, language=language, hospital_name=None))

        predicted_positive = bool((not response.fallback) and (response.answer is not None))
        expected_positive = bool(expected_benefit)
        has_notification = bool(response.notification_id and response.notification_id != "N/A")

        if response.fallback:
            fallback_count += 1

        if predicted_positive and not has_notification:
            citation_missing += 1

        matches = _answer_matches(response.answer or "", expected_benefit) if predicted_positive else False

        if predicted_positive and expected_positive and matches and has_notification:
            tp += 1
        elif predicted_positive and (not expected_positive or not matches or not has_notification):
            fp += 1
        elif (not predicted_positive) and expected_positive:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    false_positive_rate = (fp / (fp + tn)) if (fp + tn) else 0.0
    fallback_rate = (fallback_count / total) if total else 0.0

    return {
        "total": total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "citation_missing_count": citation_missing,
        "pass": false_positive_rate == 0.0,
    }


def main() -> None:
    """CLI entrypoint to run golden set validation and print report JSON."""

    parser = argparse.ArgumentParser(description="Run golden set validation for YojanaSetu agent.")
    parser.add_argument(
        "--golden-set",
        default=str(PROJECT_ROOT / "backend" / "tests" / "golden_set.json"),
        help="Path to golden_set.json file",
    )
    args = parser.parse_args()

    report = run_validation(Path(args.golden_set))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
