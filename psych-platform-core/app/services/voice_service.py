"""
Deepgram-backed streaming STT/TTS wrappers for real-time voice.

Verified directly against the installed deepgram-sdk==4.8.0 package (via
Python introspection, not just docs — see VOICE_IMPLEMENTATION_PLAN.md).
Two things worth knowing if this ever needs debugging:

1. The SDK's async event handlers must themselves be `async def` callables.
   AsyncListenWebSocketClient._emit does
   `asyncio.create_task(handler(self, *args, **kwargs))` — that only works
   if calling handler(...) returns a coroutine, i.e. handler is async.
2. `client.listen.asyncwebsocket.v("1").start(options)` connects AND spawns
   its own internal receive loop as a background task. There's no separate
   "start_listening()" call needed (unlike the sync/threaded examples in
   Deepgram's own docs, which don't apply to the async client).
"""
import asyncio
import logging
from typing import AsyncIterator, Optional, Tuple

from deepgram import (
    DeepgramClient,
    LiveOptions,
    LiveTranscriptionEvents,
    SpeakWSOptions,
    SpeakWebSocketEvents,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class TranscriptEvent:
    __slots__ = ("transcript", "is_final", "speech_final")

    def __init__(self, transcript: str, is_final: bool, speech_final: bool):
        self.transcript = transcript
        self.is_final = is_final
        self.speech_final = speech_final


# STT event type tags yielded by DeepgramSTTStream.events()
STT_TRANSCRIPT = "transcript"
STT_SPEECH_STARTED = "speech_started"
STT_UTTERANCE_END = "utterance_end"
STT_ERROR = "error"


class DeepgramSTTStream:
    """
    Wraps Deepgram's async streaming STT client behind an asyncio.Queue so
    callers can `async for event_type, payload in stream.events()` instead
    of dealing with the SDK's callback-registration model directly.

    One instance per voice call (per WebSocket connection) — not shared.
    """

    def __init__(self, sample_rate: int = 16000):
        self._client = DeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        self._conn = self._client.listen.asyncwebsocket.v("1")
        self._queue: "asyncio.Queue[Tuple[str, object]]" = asyncio.Queue()
        self._sample_rate = sample_rate
        self._started = False

        self._conn.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self._conn.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
        self._conn.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        self._conn.on(LiveTranscriptionEvents.Error, self._on_error)

    async def _on_transcript(self, _conn, result=None, **_kwargs) -> None:
        if result is None:
            return
        alt = result.channel.alternatives[0]
        if not alt.transcript:
            return
        await self._queue.put((
            STT_TRANSCRIPT,
            TranscriptEvent(
                transcript=alt.transcript,
                is_final=result.is_final,
                speech_final=result.speech_final,
            ),
        ))

    async def _on_speech_started(self, _conn, speech_started=None, **_kwargs) -> None:
        await self._queue.put((STT_SPEECH_STARTED, None))

    async def _on_utterance_end(self, _conn, utterance_end=None, **_kwargs) -> None:
        await self._queue.put((STT_UTTERANCE_END, None))

    async def _on_error(self, _conn, error=None, **_kwargs) -> None:
        logger.error("Deepgram STT error: %s", error)
        await self._queue.put((STT_ERROR, error))

    async def start(self) -> bool:
        options = LiveOptions(
            model=settings.DEEPGRAM_STT_MODEL,
            encoding="linear16",
            sample_rate=self._sample_rate,
            channels=1,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            smart_format=True,
            endpointing=300,
        )
        self._started = await self._conn.start(options)
        if not self._started:
            logger.error("DeepgramSTTStream failed to start")
        return self._started

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Feed raw 16-bit PCM mono audio at self._sample_rate."""
        if self._started:
            await self._conn.send(pcm_bytes)

    async def events(self) -> AsyncIterator[Tuple[str, object]]:
        """Yields (event_type, payload) tuples — see STT_* constants above."""
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        if self._started:
            self._started = False
            await self._conn.finish()


# TTS event type tags yielded by DeepgramTTSStream.audio_chunks()
TTS_AUDIO = "audio"
TTS_FLUSHED = "flushed"
TTS_ERROR = "error"


class DeepgramTTSStream:
    """
    Wraps Deepgram's async streaming TTS client. Feed complete sentences via
    speak(); audio chunks arrive via audio_chunks(). cancel() implements
    barge-in — it stops Deepgram's in-flight generation immediately and
    drops any audio already queued locally so playback doesn't lag behind
    an interruption with stale audio.
    """

    def __init__(self, sample_rate: int = 24000):
        self._client = DeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        self._conn = self._client.speak.asyncwebsocket.v("1")
        self._queue: "asyncio.Queue[Tuple[str, object]]" = asyncio.Queue()
        self._sample_rate = sample_rate
        self._started = False

        self._conn.on(SpeakWebSocketEvents.AudioData, self._on_audio)
        self._conn.on(SpeakWebSocketEvents.Flushed, self._on_flushed)
        self._conn.on(SpeakWebSocketEvents.Error, self._on_error)

    async def _on_audio(self, _conn, data=None, **_kwargs) -> None:
        if data:
            await self._queue.put((TTS_AUDIO, data))

    async def _on_flushed(self, _conn, flushed=None, **_kwargs) -> None:
        await self._queue.put((TTS_FLUSHED, None))

    async def _on_error(self, _conn, error=None, **_kwargs) -> None:
        logger.error("Deepgram TTS error: %s", error)
        await self._queue.put((TTS_ERROR, error))

    async def start(self) -> bool:
        options = SpeakWSOptions(
            model=settings.DEEPGRAM_TTS_MODEL,
            encoding="linear16",
            sample_rate=self._sample_rate,
        )
        self._started = await self._conn.start(options)
        if not self._started:
            logger.error("DeepgramTTSStream failed to start")
        return self._started

    async def speak(self, text: str) -> None:
        """Send one complete sentence/utterance and flush so TTS finalizes it."""
        if self._started and text.strip():
            await self._conn.send_text(text)
            await self._conn.flush()

    async def cancel(self) -> None:
        """Barge-in: stop whatever Deepgram is currently generating/queued."""
        if self._started:
            await self._conn.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        """Yields raw audio bytes as they arrive; skips control events."""
        while True:
            kind, payload = await self._queue.get()
            if kind == TTS_AUDIO:
                yield payload
            elif kind == TTS_ERROR:
                raise RuntimeError(f"Deepgram TTS error: {payload}")
            # TTS_FLUSHED is a control signal only, not yielded as audio

    async def close(self) -> None:
        if self._started:
            self._started = False
            await self._conn.finish()


def voice_enabled() -> bool:
    """Voice endpoints should 503 (not crash) if Deepgram isn't configured."""
    return bool(settings.VOICE_ENABLED and settings.DEEPGRAM_API_KEY)
