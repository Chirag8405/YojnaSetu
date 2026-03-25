"""End-to-end call orchestration for ASR, RAG query, and TTS playback."""

from __future__ import annotations

from dataclasses import dataclass
import os
import uuid
from typing import Any, Dict, Optional

from twilio.twiml.voice_response import Gather, VoiceResponse

from api.models import QueryRequest, QueryResponse
from voice.bhashini import BhashiniClient


AUDIO_STORE: Dict[str, bytes] = {}
SESSION_TRANSCRIPTS: Dict[str, str] = {}
SESSION_FAILURE_COUNT: Dict[str, int] = {}


@dataclass
class _FallbackAgent:
    """Minimal fallback agent when full backend agent has not been injected."""

    def answer(self, request: QueryRequest) -> QueryResponse:
        """Return fallback payload when runtime agent is not configured."""

        return QueryResponse(
            answer=None,
            notification_id="N/A",
            confidence=0.0,
            hospital_status="unknown",
            fallback=True,
            helpline="14555",
            language=request.language,
        )


_agent: Any = _FallbackAgent()
_bhashini_client = BhashiniClient()


def set_agent(agent: Any) -> None:
    """Inject concrete YojanaAgent instance for call orchestration."""

    global _agent
    _agent = agent


def set_bhashini_client(client: BhashiniClient) -> None:
    """Inject Bhashini client implementation for production or tests."""

    global _bhashini_client
    _bhashini_client = client


def set_session_transcript(session_id: str, transcript: str) -> None:
    """Store transcript text for a session to avoid duplicate ASR work."""

    SESSION_TRANSCRIPTS[session_id] = transcript


def _detect_language(text: str) -> str:
    """Detect Marathi vs Hindi roughly from script-specific token hints."""

    lowered = text.lower()
    marathi_markers = ["आहे", "काय", "रुग्णालय", "योजना", "मध्ये"]
    if any(marker in text for marker in marathi_markers):
        return "mr"
    if "marathi" in lowered:
        return "mr"
    return "hi"


def _audio_url(audio_id: str) -> str:
    """Build absolute audio file URL for Twilio <Play> responses."""

    base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
    return f"{base_url}/ivr/audio/{audio_id}"


def receive_call() -> str:
    """Generate TwiML greeting and speech recording prompt for incoming calls."""

    response = VoiceResponse()
    response.say("नमस्ते। यह योजना सेतु है। बीप के बाद अपना सवाल बोलिए।", language="hi-IN", voice="alice")
    response.record(
        max_length=15,
        play_beep=True,
        transcribe=True,
        transcribe_callback="/ivr/transcription",
        method="POST",
    )
    return str(response)


def process_speech(audio_url: str, session_id: str) -> str:
    """Run ASR->query->agent->TTS flow and return TwiML response for call continuation."""

    transcript = SESSION_TRANSCRIPTS.pop(session_id, "").strip()
    if not transcript:
        try:
            transcript = _bhashini_client.asr(audio_url=audio_url, language="hi")
        except Exception:
            SESSION_FAILURE_COUNT[session_id] = SESSION_FAILURE_COUNT.get(session_id, 0) + 1
            return _asr_failure_twiml(session_id)

    language = _detect_language(transcript)
    request = QueryRequest(question=transcript, language=language, hospital_name=None)
    result = _agent.answer(request)

    if result.fallback:
        return _fallback_twiml(language)

    answer_text = result.answer or "मुझे जवाब नहीं मिला।"
    tts_bytes = _bhashini_client.tts(answer_text, language)
    audio_id = str(uuid.uuid4())
    AUDIO_STORE[audio_id] = tts_bytes

    response = VoiceResponse()
    response.play(_audio_url(audio_id))
    return str(response)


def _asr_failure_twiml(session_id: str) -> str:
    """Generate DTMF fallback TwiML after repeated ASR failures."""

    response = VoiceResponse()
    attempts = SESSION_FAILURE_COUNT.get(session_id, 0)
    if attempts >= 2:
        gather = Gather(num_digits=1, action="/ivr/dtmf", method="POST")
        gather.say("क्षमा करें, हम आपकी आवाज समझ नहीं पाए। हेल्पलाइन के लिए 1 दबाएं।", language="hi-IN", voice="alice")
        response.append(gather)
        response.hangup()
        return str(response)

    response.say("कृपया फिर से बोलिए। बीप के बाद अपना सवाल दोबारा बताएं।", language="hi-IN", voice="alice")
    response.record(
        max_length=15,
        play_beep=True,
        transcribe=True,
        transcribe_callback="/ivr/transcription",
        method="POST",
    )
    return str(response)


def _fallback_twiml(language: str) -> str:
    """Generate fallback helpline TwiML with TTS plus direct dial."""

    message = (
        "माफ कीजिए, सही जानकारी अभी उपलब्ध नहीं है। सहायता के लिए 14555 पर कॉल करें।"
        if language == "hi"
        else "माफ करा, योग्य माहिती सध्या उपलब्ध नाही. मदतीसाठी 14555 वर कॉल करा."
    )
    response = VoiceResponse()
    response.say(message, language="hi-IN" if language == "hi" else "mr-IN", voice="alice")
    response.dial("14555")
    return str(response)
