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
import time
from typing import AsyncIterator, Optional, Tuple

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
    SpeakWSOptions,
    SpeakWebSocketEvents,
)

from app.core.config import settings
from app.services.voice_interface import strip_markdown_for_speech

logger = logging.getLogger(__name__)


class TranscriptEvent:
    __slots__ = ("transcript", "is_final", "speech_final")

    def __init__(self, transcript: str, is_final: bool, speech_final: bool):
        self.transcript = transcript
        self.is_final = is_final
        self.speech_final = speech_final


# Minimum spacing between STT reconnect attempts. send_audio runs ~50x/sec,
# so without this a sustained outage would hammer Deepgram with a reconnect
# per audio packet.
_STT_RECONNECT_INTERVAL_S = 2.0

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

    def __init__(self, sample_rate: int = 16000, language: str = "en"):
        # keepalive: without this, Deepgram closes the socket (code 1011) after
        # ~10-12s of no incoming audio — which happens routinely in a real call
        # (e.g. while the LLM is generating a response and the user is quiet).
        # The SDK sends periodic KeepAlive control messages in a background
        # task once this is enabled; it doesn't affect real audio/transcript flow.
        config = DeepgramClientOptions(
            api_key=settings.DEEPGRAM_API_KEY,
            options={"keepalive": "true"},
        )
        self._client = DeepgramClient(config=config)
        self._queue: "asyncio.Queue[Tuple[str, object]]" = asyncio.Queue()
        self._sample_rate = sample_rate
        self._language = language
        self._started = False
        # Distinguishes "we hung up" from "the socket dropped" — only the
        # latter should trigger a reconnect (see send_audio).
        self._closed_by_caller = False
        self._reconnect_not_before = 0.0
        self._conn = self._build_conn()

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
        # Mark the connection dead so the next send_audio() reconnects.
        # This event — not an exception from send() — is the only reliable
        # signal: the SDK catches ConnectionClosed inside its own send path
        # and logs "send() failed" rather than propagating, so send_audio()
        # sees success and _started would otherwise stay True forever.
        # Verified live 2026-08-08: a dropped socket produced 0 reconnect
        # attempts until this was added.
        self._started = False
        await self._queue.put((STT_ERROR, error))

    async def start(self) -> bool:
        options = LiveOptions(
            model=settings.DEEPGRAM_STT_MODEL,
            language=self._language,
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

    def _build_conn(self):
        """(Re)create the SDK connection object and register handlers.
        The SDK gives no way to restart a closed connection in place, so
        reconnecting means building a fresh one — hence handler registration
        lives here rather than inline in __init__."""
        conn = self._client.listen.asyncwebsocket.v("1")
        conn.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        conn.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
        conn.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        conn.on(LiveTranscriptionEvents.Error, self._on_error)
        return conn

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Feed raw 16-bit PCM mono audio at self._sample_rate.

        Transparently reconnects if Deepgram has dropped the socket. Without
        this, a single drop killed speech recognition for the REST OF THE
        CALL — the browser showed a permanent "Speech recognition error" and
        the agent never heard another word, with no path to recovery short
        of hanging up. Observed live 2026-08-07 (Deepgram close code 1011,
        "did not receive audio data within the timeout window"), triggered
        by the event-loop starvation since fixed in
        rag_service._style_exemplar_cache. Even with that root cause gone, a
        transient network blip should degrade to a brief gap in hearing, not
        a dead call."""
        if not self._started:
            if not self._closed_by_caller:
                await self._try_reconnect()
            return
        try:
            await self._conn.send(pcm_bytes)
        except Exception as exc:
            logger.warning("Deepgram STT send failed (%s) — will reconnect", exc)
            self._started = False

    async def _try_reconnect(self) -> None:
        """Rebuild the dropped connection, rate-limited so a hard outage
        can't turn into a reconnect storm (send_audio is called ~50x/sec)."""
        now = time.monotonic()
        if now < self._reconnect_not_before:
            return
        self._reconnect_not_before = now + _STT_RECONNECT_INTERVAL_S
        try:
            self._conn = self._build_conn()
            if await self.start():
                logger.info("Deepgram STT reconnected")
            else:
                logger.warning("Deepgram STT reconnect attempt failed")
        except Exception as exc:
            logger.warning("Deepgram STT reconnect error: %s", exc)

    async def events(self) -> AsyncIterator[Tuple[str, object]]:
        """Yields (event_type, payload) tuples — see STT_* constants above."""
        while True:
            yield await self._queue.get()

    async def close(self) -> None:
        self._closed_by_caller = True   # suppress reconnect on deliberate hangup
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
        # Set when Deepgram confirms it has finished generating/sending audio
        # for the most recent speak() call — see wait_until_flushed().
        self._flushed_event = asyncio.Event()

        self._conn.on(SpeakWebSocketEvents.AudioData, self._on_audio)
        self._conn.on(SpeakWebSocketEvents.Flushed, self._on_flushed)
        self._conn.on(SpeakWebSocketEvents.Error, self._on_error)

    async def _on_audio(self, _conn, data=None, **_kwargs) -> None:
        if data:
            await self._queue.put((TTS_AUDIO, data))

    async def _on_flushed(self, _conn, flushed=None, **_kwargs) -> None:
        self._flushed_event.set()
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

    async def send_text(self, text: str) -> None:
        """Queue text for synthesis WITHOUT flushing. Confirmed empirically
        (not from docs — see scratch test) that Deepgram streams audio
        continuously across multiple send_text() calls once synthesis has
        started, so calling this repeatedly across sentences (instead of
        flushing after each one) avoids an artificial restart-gap between
        them. flush() is still needed at least once to actually start
        synthesis and again at the end to drain the rest.

        Markdown is stripped here (not upstream) so it catches every
        caller — the hardcoded crisis template included "**Tele-MANAS:
        14416**", and an STT round-trip confirmed Deepgram spoke that
        literally as "Star Telemannis... Star Star"."""
        text = strip_markdown_for_speech(text)
        if self._started and text.strip():
            await self._conn.send_text(text)

    async def flush(self) -> None:
        """Force Deepgram to synthesize/send audio for everything queued so
        far. A single short sentence alone may not be enough for Deepgram to
        start synthesizing without this — confirmed empirically: no audio
        arrived after 2s with one queued sentence and no flush."""
        if self._started:
            self._flushed_event.clear()
            await self._conn.flush()

    async def speak(self, text: str) -> None:
        """Send one complete utterance and flush immediately — for single-shot
        use (e.g. the crisis path, which sends its whole message at once).
        For multi-sentence streaming responses, prefer send_text() across
        sentences plus a single flush() at the end instead — see send_text()."""
        await self.send_text(text)
        await self.flush()

    async def wait_until_flushed(self, timeout: float = 10.0) -> None:
        """Waits for Deepgram to confirm it's done generating/sending audio for
        the most recent speak() call. Callers use this to know when the AI has
        actually finished speaking (not just finished queuing text) — e.g. to
        gate barge-in correctly, since is_ai_speaking must stay true for the
        real duration of TTS output, not the few ms it takes to queue it."""
        try:
            await asyncio.wait_for(self._flushed_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Deepgram TTS flush confirmation")

    async def cancel(self) -> None:
        """Barge-in: stop whatever Deepgram is currently generating/queued."""
        if self._started:
            await self._conn.clear()
        self._flushed_event.set()
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
