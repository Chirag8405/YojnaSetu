"""Minimal i18n key catalog for API-visible strings."""

from typing import Dict


I18N_CATALOG: Dict[str, Dict[str, str]] = {
    "pmjay.eligibility.summary": {
        "hi": "pmjay.eligibility.summary",
        "mr": "pmjay.eligibility.summary",
    },
    "pmjay.documents.required": {
        "hi": "pmjay.documents.required",
        "mr": "pmjay.documents.required",
    },
}


def is_supported_i18n_key(key: str) -> bool:
    """Check if a response key is present in the catalog."""

    return key in I18N_CATALOG
