"""
WS /api/v1/ws/voice/{session_id} — real-time speech-to-speech voice calls.

Cascaded architecture (STT -> sentiment/risk gate -> RAG+LLM -> TTS), NOT
end-to-end speech-to-speech — deliberately, so the crisis safety net (see
CrisisNode / CRISIS_MESSAGE) can inspect text and force a deterministic
response before anything is spoken. See VOICE_IMPLEMENTATION_PLAN.md for
the full architecture writeup and provider rationale.

Client protocol:
  Binary frames  -> raw 16-bit PCM mono audio @ 16kHz (mic input)
  Text frames    -> reserved for future control messages (unused in V1)

Server protocol:
  Binary frames  -> raw 16-bit PCM mono audio @ 24kHz (TTS output)
  Text frames    -> JSON events:
    {"type": "ready"}
    {"type": "partial_transcript", "text": "..."}
    {"type": "final_transcript",   "text": "..."}
    {"type": "meta", "mood": "...", "risk_score": N}
    {"type": "speaking_started"}
    {"type": "speaking_ended"}
    {"type": "interrupted"}           -- barge-in: client must flush playback
    {"type": "done", "risk_level": "..."}
    {"type": "error", "message": "..."}
"""
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database import get_db
from app.infrastructure.models import ChatMessage
from app.graph.workflow import sentiment_service, rag_service
from app.services.voice_service import (
    DeepgramSTTStream,
    DeepgramTTSStream,
    voice_enabled,
    STT_TRANSCRIPT,
    STT_SPEECH_STARTED,
    STT_UTTERANCE_END,
    STT_ERROR,
)
from app.api.routes.chat import (
    CRISIS_MESSAGE,
    _get_or_create_user_session,
    _load_history,
    _load_longitudinal_context,
    _risk_level,
    _flush_sentences,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_CLINICAL_MOODS = {"anxious", "depressed", "lonely", "angry", "stressed",
                    "fearful", "hopeless", "guilty", "confused"}


class VoiceSession:
    """Coordinates STT, TTS, and barge-in state for one voice call."""

    def __init__(self, websocket: WebSocket, db: AsyncSession, user_id: str, session_id: str):
        self.ws = websocket
        self.db = db
        self.user_id = user_id
        self.session_id = session_id
        self.stt = DeepgramSTTStream(sample_rate=16000)
        self.tts = DeepgramTTSStream(sample_rate=24000)
        self.is_ai_speaking = False
        self._turn_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        stt_ok = await self.stt.start()
        tts_ok = await self.tts.start()
        return stt_ok and tts_ok

    async def send_json(self, payload: dict) -> None:
        await self.ws.send_text(json.dumps(payload))

    async def _cancel_turn_task(self) -> None:
        """Cancel and fully await the in-flight turn before starting another —
        the two must never touch self.db concurrently (AsyncSession isn't
        safe for concurrent use from multiple coroutines)."""
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass

    async def handle_barge_in(self) -> None:
        if not self.is_ai_speaking:
            return
        logger.info("Barge-in | session=%s", self.session_id)
        self.is_ai_speaking = False
        await self._cancel_turn_task()
        await self.tts.cancel()
        await self.send_json({"type": "interrupted"})

    async def pump_stt_events(self) -> None:
        """Consumes STT events; triggers barge-in and starts turns on UtteranceEnd."""
        pending_transcript = ""
        async for event_type, payload in self.stt.events():
            if event_type == STT_SPEECH_STARTED:
                await self.handle_barge_in()

            elif event_type == STT_TRANSCRIPT:
                if payload.transcript:
                    if payload.is_final:
                        pending_transcript = payload.transcript
                    await self.send_json({
                        "type": "final_transcript" if payload.is_final else "partial_transcript",
                        "text": payload.transcript,
                    })

            elif event_type == STT_UTTERANCE_END:
                text = pending_transcript.strip()
                pending_transcript = ""
                if text:
                    await self._cancel_turn_task()
                    self._turn_task = asyncio.create_task(self._run_turn(text))

            elif event_type == STT_ERROR:
                await self.send_json({"type": "error", "message": "Speech recognition error"})

    async def pump_tts_audio(self) -> None:
        """Forwards TTS audio bytes to the client as binary WS frames."""
        async for chunk in self.tts.audio_chunks():
            try:
                await self.ws.send_bytes(chunk)
            except Exception:
                return

    async def _run_turn(self, message: str) -> None:
        """One conversation turn: sentiment -> crisis-or-RAG+LLM -> TTS."""
        try:
            _, session = await _get_or_create_user_session(self.db, self.user_id, self.session_id)
            self.db.add(ChatMessage(session_id=session.id, role="user", content=message))
            await self.db.commit()

            longitudinal_context = await _load_longitudinal_context(
                self.db, self.user_id, self.session_id
            )
            analysis, history = await asyncio.gather(
                sentiment_service.analyze_sentiment(message),
                _load_history(self.db, self.session_id),
            )
            mood = analysis.get("mood", "neutral")
            risk_score = int(analysis.get("risk_score", 0))
            is_crisis = risk_score >= 8
            level = _risk_level(risk_score)

            await self.send_json({"type": "meta", "mood": mood, "risk_score": risk_score})

            full_response = ""
            self.is_ai_speaking = True
            await self.send_json({"type": "speaking_started"})

            if is_crisis:
                full_response = CRISIS_MESSAGE
                await self.tts.speak(CRISIS_MESSAGE)
            else:
                if len(message.split()) >= 4 or mood in _CLINICAL_MOODS:
                    context = await rag_service.retrieve_clinical_context(message, mood)
                else:
                    context = ""
                if longitudinal_context and context:
                    context = f"[Previous sessions]\n{longitudinal_context}\n\n[Clinical evidence]\n{context}"
                elif longitudinal_context:
                    context = f"[Previous sessions]\n{longitudinal_context}"

                history.append({"role": "user", "content": message})

                buf = ""
                async for token in sentiment_service.stream_therapeutic_response(
                    history=history, context=context, mood=mood
                ):
                    full_response += token
                    buf += token
                    sentences, buf = _flush_sentences(buf)
                    for sentence in sentences:
                        await self.tts.speak(sentence)
                if buf.strip():
                    await self.tts.speak(buf.strip())

            self.is_ai_speaking = False
            await self.send_json({"type": "speaking_ended"})

            self.db.add(ChatMessage(
                session_id=session.id, role="assistant", content=full_response, detected_mood=mood,
            ))
            if level != "LOW":
                session.risk_level = level
                self.db.add(session)
            await self.db.commit()

            await self.send_json({"type": "done", "risk_level": level})

        except asyncio.CancelledError:
            self.is_ai_speaking = False
            raise
        except Exception as e:
            logger.error("Voice turn error | session=%s | %s", self.session_id, str(e), exc_info=True)
            self.is_ai_speaking = False
            await self.send_json({"type": "error", "message": "Processing failed"})

    async def close(self) -> None:
        await self._cancel_turn_task()
        await self.stt.close()
        await self.tts.close()


@router.websocket("/ws/voice/{session_id}")
async def websocket_voice(
    websocket: WebSocket,
    session_id: str,
    token: str = "",
    db: AsyncSession = Depends(get_db),
):
    """
    Auth mirrors /ws/chat: browsers can't set an Authorization header on the
    WS handshake, so the JWT is passed as a query param.
    """
    from jose import JWTError
    from app.core.security import decode_token

    if not voice_enabled():
        await websocket.accept()
        await websocket.close(code=4503, reason="Voice is not configured on this server")
        return

    try:
        authenticated_user_id = decode_token(token)
    except JWTError:
        await websocket.accept()
        await websocket.close(code=4401, reason="Invalid or missing token")
        return

    await websocket.accept()
    logger.info("Voice WebSocket connected | session=%s | user=%s", session_id, authenticated_user_id)

    voice_session = VoiceSession(websocket, db, authenticated_user_id, session_id)
    if not await voice_session.start():
        await websocket.send_text(json.dumps({"type": "error", "message": "Voice provider unavailable"}))
        await websocket.close(code=4502, reason="Could not start voice provider")
        return

    stt_task = asyncio.create_task(voice_session.pump_stt_events())
    tts_task = asyncio.create_task(voice_session.pump_tts_audio())
    await voice_session.send_json({"type": "ready"})

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await voice_session.stt.send_audio(message["bytes"])
            # Text frames reserved for future control messages (e.g. mute) — ignored in V1.
    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected | session=%s", session_id)
    finally:
        stt_task.cancel()
        tts_task.cancel()
        await voice_session.close()
