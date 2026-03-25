"""Tests for IVR voice integration flow and Twilio webhooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from fastapi.testclient import TestClient

from api import main as api_main
from api.models import QueryResponse
from voice import call_flow
from voice.bhashini import BhashiniClient


@dataclass
class _FakeAgent:
    """Deterministic fake agent for voice flow tests."""

    fallback: bool = False
    received_questions: List[str] | None = None

    def answer(self, request: Any) -> QueryResponse:
        """Return deterministic response and capture asked question."""

        if self.received_questions is None:
            self.received_questions = []
        self.received_questions.append(request.question)

        if self.fallback:
            return QueryResponse(
                answer=None,
                notification_id="N/A",
                confidence=0.2,
                hospital_status="unknown",
                fallback=True,
                helpline="14555",
                language=request.language,
            )

        return QueryResponse(
            answer="PMJAY covers this. notification_id: PMJAY-MH-2024-701",
            notification_id="PMJAY-MH-2024-701",
            confidence=0.9,
            hospital_status="unknown",
            fallback=False,
            helpline=None,
            language=request.language,
        )


class _FakeBhashini(BhashiniClient):
    """Fake Bhashini client returning deterministic ASR/TTS values."""

    def __init__(self) -> None:
        """Create fake client without external API requirements."""

        super().__init__(base_url="http://fake", api_key="fake")

    def asr(self, audio_url: str, language: str) -> str:
        """Return deterministic transcript for ASR test flow."""

        _ = (audio_url, language)
        return "Is PMJAY available for surgery?"

    def tts(self, text: str, language: str) -> bytes:
        """Return fake MP3 byte content for TwiML play flow."""

        _ = (text, language)
        return b"FAKE_MP3"


def test_incoming_call_returns_greeting_twiml() -> None:
    """Return valid TwiML greeting and recording prompt for incoming calls."""

    client = TestClient(api_main.app)
    response = client.post("/ivr/incoming")

    assert response.status_code == 200
    assert "<Response>" in response.text
    assert "<Record" in response.text


def test_transcription_flow_sends_query_to_agent() -> None:
    """Use transcription callback flow and ensure question is sent to agent."""

    fake_agent = _FakeAgent(fallback=False)
    call_flow.set_agent(fake_agent)
    call_flow.set_bhashini_client(_FakeBhashini())

    client = TestClient(api_main.app)
    response = client.post(
        "/ivr/transcription",
        data={
            "CallSid": "CA_TEST_1",
            "RecordingUrl": "https://example.com/audio1.mp3",
            "TranscriptionText": "",
            "TranscriptionConfidence": "0.2",
        },
    )

    assert response.status_code == 200
    assert fake_agent.received_questions is not None
    assert len(fake_agent.received_questions) == 1
    assert "<Play>" in response.text


def test_fallback_response_returns_helpline_twiml() -> None:
    """Return helpline Dial TwiML when agent marks response as fallback."""

    fake_agent = _FakeAgent(fallback=True)
    call_flow.set_agent(fake_agent)
    call_flow.set_bhashini_client(_FakeBhashini())

    client = TestClient(api_main.app)
    response = client.post(
        "/ivr/transcription",
        data={
            "CallSid": "CA_TEST_2",
            "RecordingUrl": "https://example.com/audio2.mp3",
            "TranscriptionText": "",
            "TranscriptionConfidence": "0.2",
        },
    )

    assert response.status_code == 200
    assert "<Dial>14555</Dial>" in response.text
