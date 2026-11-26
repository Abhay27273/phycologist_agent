"""
Per-reply language routing for voice calls.

A call can switch language mid-conversation (the way ChatGPT's and Claude's
voice modes follow whichever language the person speaks). Speech-to-text is
handled by one auto-detecting stream; this wrapper does the output side: each
reply is spoken by the engine that suits its language.

    English          -> `english` engine (Deepgram Aura: lowest latency)
    Hindi / Hinglish -> `indic` engine   (Sarvam Bulbul: native Hindi prosody)

The voice loop already knows the language of the turn before it speaks (it is
detected from the transcript), so it calls set_language() once per reply and a
reply is never split across two voices. Both engines output 16-bit PCM mono at
the same playback rate, so their audio is interchangeable on the client.

Either engine may be missing (key not configured); the other covers everything.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from app.services.voice_interface import TTSStream

logger = logging.getLogger(__name__)

INDIC_LANGUAGES = ("hi", "hinglish")


class LanguageRoutedTTS(TTSStream):
    def __init__(self, english: Optional[TTSStream], indic: Optional[TTSStream]):
        if english is None and indic is None:
            raise ValueError("LanguageRoutedTTS needs at least one engine")
        self._english = english
        self._indic = indic
        self._active: TTSStream = english or indic  # type: ignore[assignment]
        self._started: list[TTSStream] = []
        self._merged: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._forwarders: list[asyncio.Task] = []

    # -- routing ---------------------------------------------------------

    def set_language(self, language: str) -> None:
        """Choose the engine for the next reply. Call before speaking."""
        want_indic = language in INDIC_LANGUAGES
        if want_indic and self._indic in self._started:
            self._active = self._indic
        elif not want_indic and self._english in self._started:
            self._active = self._english
        elif self._started:
            self._active = self._started[0]  # preferred engine is down: use what works

    @property
    def active_name(self) -> str:
        return "indic" if self._active is self._indic else "english"

    # -- TTSStream -------------------------------------------------------

    async def start(self) -> bool:
        for engine in (self._english, self._indic):
            if engine is None:
                continue
            try:
                if await engine.start():
                    self._started.append(engine)
            except Exception as exc:  # one engine failing must not take the call down
                logger.error("TTS engine failed to start: %s", type(exc).__name__)
        if not self._started:
            return False
        self.set_language("en")
        return True

    async def send_text(self, text: str) -> None:
        await self._active.send_text(text)

    async def flush(self) -> None:
        await self._active.flush()

    async def speak(self, text: str) -> None:
        await self._active.speak(text)

    async def wait_until_flushed(self, timeout: float = 10.0) -> None:
        await self._active.wait_until_flushed(timeout)

    async def cancel(self) -> None:
        # Barge-in: whichever engine was talking must stop; cancelling both is
        # cheap and covers a language switch racing an interruption.
        for engine in self._started:
            try:
                await engine.cancel()
            except Exception as exc:
                logger.warning("TTS cancel failed: %s", type(exc).__name__)
        # Drop audio already merged but not yet sent, or it plays after the interruption.
        while not self._merged.empty():
            try:
                self._merged.get_nowait()
            except asyncio.QueueEmpty:
                break

    def audio_chunks(self) -> AsyncIterator[bytes]:
        return self._audio_generator()

    async def _forward(self, engine: TTSStream) -> None:
        """Pump one engine's audio into the shared queue. A failed synthesis
        request is logged and reading resumes, so one bad request does not end
        the call's audio for good (a raising generator is finished)."""
        while True:
            try:
                async for chunk in engine.audio_chunks():
                    await self._merged.put(chunk)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TTS stream error, continuing: %s", str(exc)[:120])
                await asyncio.sleep(0.05)

    async def _audio_generator(self):
        if not self._forwarders:
            self._forwarders = [asyncio.create_task(self._forward(e)) for e in self._started]
        while True:
            yield await self._merged.get()

    async def close(self) -> None:
        for task in self._forwarders:
            task.cancel()
        for task in self._forwarders:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._forwarders = []
        for engine in self._started:
            try:
                await engine.close()
            except Exception as exc:
                logger.warning("TTS close failed: %s", type(exc).__name__)
