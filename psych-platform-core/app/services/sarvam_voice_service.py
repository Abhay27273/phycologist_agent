"""
Sarvam AI streaming STT/TTS wrappers — Phase 3.

Implements the same STTStream / TTSStream interface as Deepgram so that
VoiceSession can swap providers transparently based on session language.

WHY Sarvam for Hindi/Hinglish sessions:
  - Saaras v3 mode=codemix handles natural Hinglish code-switching natively;
    global STT models show 30-50% relative WER increase on code-switched speech.
  - Bulbul v3 renders Hinglish TTS in a single pass — no language-boundary
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
import base64
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
    Sarvam Saaras v3 streaming STT via WebSocket (wss://api.sarvam.ai/speech-to-text/ws).

    Accepts 16-bit PCM mono @ 16kHz - identical to AudioWorklet output.
    Uses mode=codemix for natural Hinglish transcription.

    Protocol (verified against the live API, 2026-11-26): audio goes up as JSON
    {"audio": {"data": <base64 pcm>, "sample_rate": 16000, "encoding": "audio/wav"}};
    server-side VAD sends {"type":"events","data":{"signal_type":"START_SPEECH"|"END_SPEECH"}}
    and, once per finished utterance, {"type":"data","data":{"transcript": ...}}.
    There are no interim transcripts, so each transcript is reported as a final,
    speech_final segment and the voice loop's semantic/stall checks decide the turn.
    """

    SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"

    def __init__(self, sample_rate: int = 16000, language_code: str = "hi-IN"):
        self._sample_rate = sample_rate
        self._language_code = language_code
        self._queue: asyncio.Queue[Tuple[str, object]] = asyncio.Queue()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._started = False
        self._closing = False
        self._reconnect_lock = asyncio.Lock()

    def _url(self) -> str:
        from urllib.parse import urlencode
        q = urlencode({
            "model": settings.SARVAM_STT_MODEL,
            "language-code": self._language_code,
            "mode": "codemix",
            "sample_rate": self._sample_rate,
            "input_audio_codec": "pcm_s16le",
            "vad_signals": "true",
        })
        return f"{self.SARVAM_WS_URL}?{q}"

    async def _connect(self) -> bool:
        import websockets
        headers = {"Api-Subscription-Key": settings.SARVAM_API_KEY}
        try:
            self._ws = await websockets.connect(
                self._url(), additional_headers=headers, ping_interval=20, open_timeout=10,
            )
        except TypeError:  # websockets < 14 spells it extra_headers
            self._ws = await websockets.connect(
                self._url(), extra_headers=headers, ping_interval=20, open_timeout=10,
            )
        self._recv_task = asyncio.create_task(self._receive_loop(self._ws))
        return True

    async def start(self) -> bool:
        if not sarvam_enabled():
            logger.error("SarvamSTTStream: SARVAM_API_KEY not configured")
            return False
        try:
            await self._connect()
            self._started = True
            logger.info("SarvamSTTStream started | model=%s lang=%s", settings.SARVAM_STT_MODEL, self._language_code)
            return True
        except Exception as e:
            logger.error("SarvamSTTStream failed to start: %s", type(e).__name__)
            return False

    async def _receive_loop(self, ws) -> None:
        """Read JSON responses from Sarvam and translate to STT_* events."""
        try:
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg) if isinstance(raw_msg, (str, bytes)) else {}
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type", "")
                data = msg.get("data") or {}

                if kind == "events":
                    if data.get("signal_type") == "START_SPEECH":
                        await self._queue.put((STT_SPEECH_STARTED, None))
                elif kind == "data":
                    transcript = (data.get("transcript") or "").strip()
                    if transcript:
                        await self._queue.put((
                            STT_TRANSCRIPT,
                            TranscriptEvent(transcript=transcript, is_final=True, speech_final=True),
                        ))
                elif kind == "error":
                    detail = data.get("message") or data.get("error") or "Unknown Sarvam error"
                    logger.error("Sarvam STT error: %s", str(detail)[:200])
                    await self._queue.put((STT_ERROR, str(detail)))
        except Exception as e:
            if self._started and not self._closing:
                logger.warning("SarvamSTTStream receive loop ended: %s", type(e).__name__)
                await self._queue.put((STT_ERROR, type(e).__name__))
        finally:
            # Mark the socket dead so the next audio packet reconnects.
            if self._ws is ws:
                self._ws = None

    async def _ensure_connected(self) -> bool:
        if self._ws is not None:
            return True
        async with self._reconnect_lock:
            if self._ws is not None:
                return True
            try:
                await self._connect()
                logger.info("SarvamSTTStream reconnected")
                return True
            except Exception as e:
                logger.warning("SarvamSTTStream reconnect failed: %s", type(e).__name__)
                await asyncio.sleep(1.0)  # don't hammer on every 20 ms packet
                return False

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if not self._started or self._closing:
            return
        if not await self._ensure_connected():
            return
        try:
            await self._ws.send(json.dumps({
                "audio": {
                    "data": base64.b64encode(pcm_bytes).decode("ascii"),
                    "sample_rate": self._sample_rate,
                    "encoding": "audio/wav",
                }
            }))
        except Exception as e:
            logger.warning("SarvamSTTStream send_audio error: %s", type(e).__name__)
            self._ws = None

    def events(self) -> AsyncIterator[Tuple[str, object]]:
        return self._events_generator()

    async def _events_generator(self):
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        self._closing = True
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
            self._ws = None


class SarvamTTSStream(TTSStream):
    """
    Sarvam Bulbul v3 streaming TTS.

    Handles Hinglish/Tanglish code-switching in a single pass — no language
    boundary detection or engine routing required.

    Implementation: uses Sarvam's REST TTS API with chunked streaming response.
    Audio is requested as 16-bit PCM mono @ the playback rate (24 kHz) via speech_sample_rate.
    """

    SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, sample_rate: int = 24000):
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[Tuple[str, object]] = asyncio.Queue()
        self._started = False
        self._flushed_event = asyncio.Event()
        self._text_buffer: list[str] = []
        # Synthesis jobs run strictly one at a time, in order. The voice loop
        # flushes twice per reply (after the first sentence, then at the end);
        # running those concurrently interleaved audio out of order and let the
        # first job to finish release wait_until_flushed() while the second was
        # still producing audio.
        self._jobs: asyncio.Queue[str] = asyncio.Queue()
        self._pending = 0
        self._worker_task: asyncio.Task | None = None

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
            # Sarvam's REST TTS only returns audio once a whole request is
            # synthesized, so waiting for flush() to batch the reply would delay
            # everything after the first sentence until the LLM is done. Each
            # send_text() is one sentence from the voice loop: queue it now and
            # let the worker synthesize it (in order) while the previous
            # sentence is still playing.
            await self.flush()

    async def flush(self) -> None:
        if not self._started or not self._text_buffer:
            return
        text_to_speak = " ".join(self._text_buffer)
        self._text_buffer.clear()
        self._pending += 1
        self._flushed_event.clear()
        self._jobs.put_nowait(text_to_speak)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            text = await self._jobs.get()
            try:
                await self._synthesize(text)
            except asyncio.CancelledError:
                raise
            finally:
                self._pending = max(0, self._pending - 1)
                if self._pending == 0:
                    self._flushed_event.set()
                    await self._queue.put((TTS_FLUSHED, None))

    async def speak(self, text: str) -> None:
        await self.send_text(text)
        await self.flush()

    async def _synthesize(self, text: str) -> None:
        """Call Sarvam TTS REST API and stream audio chunks to queue."""
        try:
            import httpx
            headers = {
                "api-subscription-key": settings.SARVAM_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "target_language_code": "hi-IN",
                "speaker": settings.SARVAM_TTS_VOICE,
                "model": settings.SARVAM_TTS_MODEL,
                "speech_sample_rate": self._sample_rate,
                "enable_preprocessing": True,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", self.SARVAM_TTS_URL, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error("Sarvam TTS error %d: %s", response.status_code, error_text[:200].decode("utf-8", "replace"))
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

    async def wait_until_flushed(self, timeout: float = 10.0) -> None:
        # Sentence-by-sentence synthesis is queued behind the LLM stream, so the
        # tail of a long (guidance) reply legitimately takes longer than the
        # 10 s Deepgram-era default.
        timeout = max(timeout, 30.0)
        try:
            await asyncio.wait_for(self._flushed_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Sarvam TTS flush confirmation")

    async def cancel(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        while not self._jobs.empty():
            try:
                self._jobs.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._pending = 0
        self._text_buffer.clear()
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
