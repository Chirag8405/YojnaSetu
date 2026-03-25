"""Twilio IVR webhook routes for incoming calls and transcriptions."""

from __future__ import annotations

from fastapi import APIRouter, Form, Response

from voice import call_flow


router = APIRouter(prefix="/ivr", tags=["ivr"])


@router.post("/incoming")
def incoming_call() -> Response:
    """Handle Twilio incoming webhook and return initial greeting TwiML."""

    twiml = call_flow.receive_call()
    return Response(content=twiml, media_type="application/xml")


@router.post("/transcription")
def transcription_callback(
    CallSid: str = Form(default=""),
    RecordingUrl: str = Form(default=""),
    TranscriptionText: str = Form(default=""),
    TranscriptionConfidence: float = Form(default=0.0),
) -> Response:
    """Handle Twilio transcription and fallback to Bhashini ASR below confidence threshold."""

    session_id = CallSid or "anonymous-session"
    if TranscriptionText.strip() and float(TranscriptionConfidence) >= 0.8:
        call_flow.set_session_transcript(session_id=session_id, transcript=TranscriptionText.strip())

    twiml = call_flow.process_speech(audio_url=RecordingUrl, session_id=session_id)
    return Response(content=twiml, media_type="application/xml")


@router.post("/dtmf")
def dtmf_handler(Digits: str = Form(default="")) -> Response:
    """Handle DTMF fallback for routing users to helpline."""

    if Digits.strip() == "1":
        xml = "<Response><Dial>14555</Dial></Response>"
    else:
        xml = "<Response><Say language='hi-IN'>कृपया दोबारा प्रयास करें।</Say><Hangup/></Response>"
    return Response(content=xml, media_type="application/xml")


@router.get("/audio/{audio_id}")
def get_audio(audio_id: str) -> Response:
    """Serve generated TTS audio bytes for Twilio <Play> playback."""

    audio_bytes = call_flow.AUDIO_STORE.get(audio_id, b"")
    status_code = 200 if audio_bytes else 404
    return Response(content=audio_bytes, media_type="audio/mpeg", status_code=status_code)
