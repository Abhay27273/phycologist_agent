"""
Abstract voice provider interface — Phase 3.

Decouples VoiceSession in routes/voice.py from Deepgram specifics so that
SarvamSTTStream / SarvamTTSStream can be swapped in behind the same interface
when the user's session language is "hi" or "hinglish".

The cascaded safety architecture (STT → sentiment risk gate → LLM → TTS) is
vendor-independent — the text checkpoint holds regardless of provider.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import AsyncIterator, Tuple

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def strip_markdown_for_speech(text: str) -> str:
    """TTS engines vocalize markdown syntax literally rather than treating it
    as formatting — confirmed empirically via an STT round-trip: the crisis
    message's "**Tele-MANAS: 14416**" was actually spoken as "Star
    Telemannis... Star Star, also...". Applied at both concrete TTS
    implementations' send_text()/speak() so every caller gets this for
    free, including the hardcoded crisis template (the single most
    safety-critical text in the system) and any future call site — not
    just today's known offenders."""
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_BOLD_UNDERSCORE_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    # Defensive: drop any unpaired markdown markers left over rather than
    # let TTS vocalize them literally.
    return text.replace("**", "").replace("__", "")


class STTStream(ABC):
    """Async streaming speech-to-text provider interface."""

    @abstractmethod
    async def start(self) -> bool:
        """Connect to the provider. Returns True on success."""

    @abstractmethod
    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Feed raw 16-bit PCM mono audio at 16 kHz."""

    @abstractmethod
    def events(self) -> AsyncIterator[Tuple[str, object]]:
        """
        Async iterator yielding (event_type, payload) tuples.
        event_type must be one of the STT_* constants from voice_service.py:
          STT_TRANSCRIPT, STT_SPEECH_STARTED, STT_UTTERANCE_END, STT_ERROR
        """

    @abstractmethod
    async def close(self) -> None:
        """Disconnect and release resources."""


class TTSStream(ABC):
    """Async streaming text-to-speech provider interface."""

    @abstractmethod
    async def start(self) -> bool:
        """Connect to the provider. Returns True on success."""

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Queue text for synthesis without flushing. For multi-sentence streaming."""

    @abstractmethod
    async def flush(self) -> None:
        """Force synthesis to start for all queued text."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Send one complete utterance and flush immediately (crisis path / single-shot)."""

    @abstractmethod
    async def wait_until_flushed(self, timeout: float = 10.0) -> None:
        """Block until Deepgram/Sarvam confirms it has finished generating audio."""

    @abstractmethod
    async def cancel(self) -> None:
        """Barge-in: stop generation and drop queued audio."""

    @abstractmethod
    def audio_chunks(self) -> AsyncIterator[bytes]:
        """Async iterator yielding raw audio bytes as they arrive."""

    @abstractmethod
    async def close(self) -> None:
        """Disconnect and release resources."""
