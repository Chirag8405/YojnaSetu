"""Bhashini ASR/TTS client wrapper with in-memory TTS caching."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Dict, Tuple

import httpx


SUPPORTED_LANGUAGES = {"hi", "mr"}


@dataclass
class BhashiniClient:
    """Client for Bhashini speech-to-text and text-to-speech APIs."""

    base_url: str = field(default_factory=lambda: os.getenv("BHASHINI_BASE_URL", "").rstrip("/"))
    api_key: str = field(default_factory=lambda: os.getenv("BHASHINI_API_KEY", ""))
    timeout_seconds: float = 20.0
    _tts_cache: Dict[Tuple[str, str], bytes] = field(default_factory=dict)

    def _validate_language(self, language: str) -> None:
        """Ensure language code is among supported Hindi/Marathi options."""

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

    def asr(self, audio_url: str, language: str) -> str:
        """Convert speech audio URL to text via Bhashini ASR endpoint."""

        self._validate_language(language)
        if not self.base_url or not self.api_key:
            raise RuntimeError("Bhashini ASR unavailable: missing BHASHINI_BASE_URL or BHASHINI_API_KEY.")

        payload = {"audio_url": audio_url, "language": language}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/asr", json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()

        text = str(body.get("text", "")).strip()
        if not text:
            raise RuntimeError("Bhashini ASR returned empty text.")
        return text

    def tts(self, text: str, language: str) -> bytes:
        """Convert text to MP3 bytes with cache reuse for identical inputs."""

        self._validate_language(language)
        cache_key = (text.strip(), language)
        if cache_key in self._tts_cache:
            return self._tts_cache[cache_key]

        if not self.base_url or not self.api_key:
            raise RuntimeError("Bhashini TTS unavailable: missing BHASHINI_BASE_URL or BHASHINI_API_KEY.")

        payload = {"text": text, "language": language, "format": "mp3"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/tts", json=payload, headers=headers)
            response.raise_for_status()

        audio_bytes = response.content
        if not audio_bytes:
            raise RuntimeError("Bhashini TTS returned empty audio payload.")

        self._tts_cache[cache_key] = audio_bytes
        return audio_bytes
