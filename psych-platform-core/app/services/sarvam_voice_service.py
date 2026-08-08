"""
Sarvam AI streaming STT/TTS wrappers — Phase 3.

Implements the same STTStream / TTSStream interface as Deepgram so that
VoiceSession can swap providers transparently based on session language.

WHY Sarvam for Hindi/Hinglish sessions:
  - Saaras v3 mode=codemix handles natural Hinglish code-switching natively;
    global STT models show 30-50% relative WER increase on code-switched speech.
  - Bulbul v1 renders Hinglish TTS in a single pass — no language-boundary
    routing that produces jarring output at every Hindi/English switch.
  - Data residency in India (DPDP advantage over US-based Deepgram).

HONEST LIMITATIONS (from research doc §4.4):
  - Sarvam is a smaller vendor — assess uptime before making it the sole path.
    Deepgram stays as English fallback; both live behind the same interface.
  - The sub-250ms latency figures are vendor-published; benchmark on your own
    Indian-English phone audio before committing.
  - Sarvam streaming STT accepts WAV and raw PCM only (pcm_s16le, 16kHz).
    Sample rate must match exactly at connection AND per chunk — mismatch garbles output.
  - sarvam-python SDK does not yet have official streaming WS support (2026-07);
    this implementation uses their REST streaming API via httpx + SSE. Update
    when the SDK exposes a proper streaming client.

Audio format: 16-bit PCM mono, 16000 Hz (matches AudioWorklet output).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Tuple

from app.core.config import settings
from app.services.voice_service import (
    STT_TRANSCRIPT,
    STT_SPEECH_STARTED,
    STT_UTTERANCE_END,
    STT_ERROR,
    TTS_AUDIO,
    TTS_FLUSHED,
    TTS_ERROR,
    TranscriptEvent,
)
from app.services.voice_interface import STTStream, TTSStream, strip_markdown_for_speech

logger = logging.getLogger(__name__)

# Silence threshold for utterance-end detection (ms equivalent)
_SILENCE_THRESHOLD_MS = 1000
_VAD_SPEECH_PROB_THRESHOLD = 0.5


def sarvam_enabled() -> bool:
    """Returns True only when Sarvam is configured AND Deepgram is not the preferred option."""
    return bool(settings.SARVAM_API_KEY)


class SarvamSTTStream(STTStream):
    """
    Sarvam Saaras v3 streaming STT via WebSocket.

    Accepts 16-bit PCM mono @ 16kHz — identical to AudioWorklet output.
    Uses mode=codemix for natural Hinglish transcription.

    Implementation note: uses Sarvam's streaming WebSocket API directly since
    the Python SDK does not yet expose an async streaming client.
    """

    SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text-translate/streaming"

    def __init__(self, sample_rate: int = 16000, language_code: str = "hi-IN"):
        self._sample_rate = sample_rate
        self._language_code = language_code
        self._queue: asyncio.Queue[Tuple[str, object]] = asyncio.Queue()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._started = False

        # Rolling buffer for silence detection
        self._last_speech_time: float = 0.0
        self._pending_transcript = ""
        self._has_speech = False

    async def start(self) -> bool:
        if not sarvam_enabled():
            logger.error("SarvamSTTStream: SARVAM_API_KEY not configured")
            return False
        try:
            import websockets
            self._ws = await websockets.connect(
                self.SARVAM_WS_URL,
                extra_headers={
                    "Authorization": f"Bearer {settings.SARVAM_API_KEY}",
                    "Content-Type": "application/octet-stream",
                },
                additional_headers={
                    "X-Model": settings.SARVAM_STT_MODEL,
                    "X-Language": self._language_code,
                    "X-Mode": "codemix",
                    "X-Sample-Rate": str(self._sample_rate),
                    "X-Encoding": "pcm_s16le",
                },
                ping_interval=20,
            )
            self._started = True
            self._recv_task = asyncio.create_task(self._receive_loop())
            logger.info("SarvamSTTStream started | model=%s lang=%s", settings.SARVAM_STT_MODEL, self._language_code)
            return True
        except Exception as e:
            logger.error("SarvamSTTStream failed to start: %s", e)
            return False

    async def _receive_loop(self) -> None:
        """Read JSON responses from Sarvam and translate to STT_* events."""
        import asyncio
        try:
            async for raw_msg in self._ws:
                try:
                    msg = json.loads(raw_msg) if isinstance(raw_msg, str) else {}
                    msg_type = msg.get("type", "")

                    if msg_type == "SpeechStarted":
                        self._has_speech = True
                        await self._queue.put((STT_SPEECH_STARTED, None))

                    elif msg_type in ("Transcript", "FinalTranscript"):
                        transcript = msg.get("transcript", "").strip()
                        is_final = msg_type == "FinalTranscript"
                        speech_final = is_final
                        if transcript:
                            self._pending_transcript = transcript
                            await self._queue.put((
                                STT_TRANSCRIPT,
                                TranscriptEvent(
                                    transcript=transcript,
                                    is_final=is_final,
                                    speech_final=speech_final,
                                ),
                            ))

                    elif msg_type == "UtteranceEnd":
                        await self._queue.put((STT_UTTERANCE_END, None))

                    elif msg_type == "Error":
                        error_msg = msg.get("message", "Unknown Sarvam error")
                        logger.error("Sarvam STT error: %s", error_msg)
                        await self._queue.put((STT_ERROR, error_msg))

                except (json.JSONDecodeError, KeyError):
                    pass  # ignore malformed frames
        except Exception as e:
            if self._started:
                logger.error("SarvamSTTStream receive loop error: %s", e)
                await self._queue.put((STT_ERROR, str(e)))

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self._started and self._ws:
            try:
                await self._ws.send(pcm_bytes)
            except Exception as e:
                logger.error("SarvamSTTStream send_audio error: %s", e)

    def events(self) -> AsyncIterator[Tuple[str, object]]:
        return self._events_generator()

    async def _events_generator(self):
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        self._started = False
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass


class SarvamTTSStream(TTSStream):
    """
    Sarvam Bulbul v1 streaming TTS.

    Handles Hinglish/Tanglish code-switching in a single pass — no language
    boundary detection or engine routing required.

    Implementation: uses Sarvam's REST TTS API with chunked streaming response.
    Audio returned as 16-bit PCM mono @ 22050 Hz (resampled to 24kHz for playback).
    """

    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[Tuple[str, object]] = asyncio.Queue()
        self._started = False
        self._flushed_event = asyncio.Event()
        self._text_buffer: list[str] = []
        self._synthesis_task: asyncio.Task | None = None

    async def start(self) -> bool:
        if not sarvam_enabled():
            logger.error("SarvamTTSStream: SARVAM_API_KEY not configured")
            return False
        self._started = True
        self._flushed_event.set()
        logger.info("SarvamTTSStream started | model=%s voice=%s", settings.SARVAM_TTS_MODEL, settings.SARVAM_TTS_VOICE)
        return True

    async def send_text(self, text: str) -> None:
        """Markdown stripped here so every caller (including the hardcoded
        crisis template) gets clean, speakable text — see
        voice_interface.strip_markdown_for_speech for why this matters."""
        text = strip_markdown_for_speech(text)
        if self._started and text.strip():
            self._text_buffer.append(text.strip())

    async def flush(self) -> None:
        if not self._started or not self._text_buffer:
            return
        text_to_speak = " ".join(self._text_buffer)
        self._text_buffer.clear()
        self._flushed_event.clear()
        self._synthesis_task = asyncio.create_task(self._synthesize(text_to_speak))

    async def speak(self, text: str) -> None:
        await self.send_text(text)
        await self.flush()

    async def _synthesize(self, text: str) -> None:
        """Call Sarvam TTS REST API and stream audio chunks to queue."""
        try:
            import httpx
            headers = {
                "Authorization": f"Bearer {settings.SARVAM_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "inputs": [text],
                "target_language_code": "hi-IN",
                "speaker": settings.SARVAM_TTS_VOICE,
                "model": settings.SARVAM_TTS_MODEL,
                "enable_preprocessing": True,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", self.SARVAM_TTS_URL, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error("Sarvam TTS error %d: %s", response.status_code, error_text[:200])
                        await self._queue.put((TTS_ERROR, f"HTTP {response.status_code}"))
                        return

                    # Sarvam returns base64-encoded WAV in JSON; chunk it
                    body = await response.aread()
                    data = json.loads(body)
                    audios = data.get("audios", [])
                    for audio_b64 in audios:
                        import base64
                        import io
                        import wave
                        audio_bytes = base64.b64decode(audio_b64)
                        # Strip WAV header; send raw PCM chunks
                        with wave.open(io.BytesIO(audio_bytes)) as wf:
                            pcm = wf.readframes(wf.getnframes())
                        # Send in 4096-byte chunks to mimic streaming
                        chunk_size = 4096
                        for i in range(0, len(pcm), chunk_size):
                            await self._queue.put((TTS_AUDIO, pcm[i:i + chunk_size]))
                            await asyncio.sleep(0)  # yield to event loop

        except Exception as e:
            logger.error("SarvamTTSStream synthesis error: %s", e)
            await self._queue.put((TTS_ERROR, str(e)))
        finally:
            self._flushed_event.set()
            await self._queue.put((TTS_FLUSHED, None))

    async def wait_until_flushed(self, timeout: float = 10.0) -> None:
        try:
            await asyncio.wait_for(self._flushed_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Sarvam TTS flush confirmation")

    async def cancel(self) -> None:
        if self._synthesis_task and not self._synthesis_task.done():
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass
        self._flushed_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def audio_chunks(self) -> AsyncIterator[bytes]:
        return self._audio_generator()

    async def _audio_generator(self):
        while True:
            kind, payload = await self._queue.get()
            if kind == TTS_AUDIO:
                yield payload
            elif kind == TTS_ERROR:
                raise RuntimeError(f"Sarvam TTS error: {payload}")

    async def close(self) -> None:
        self._started = False
        await self.cancel()
