"""
A reply is flushed twice (after its first sentence, then at the end). A late
confirmation of the FIRST flush must not mark the whole reply as finished.
"""
import asyncio

import pytest

from app.services.voice_service import DeepgramTTSStream


class FakeConn:
    def __init__(self):
        self.sent, self.flushes, self.cleared = [], 0, 0

    async def send_text(self, t):
        self.sent.append(t)

    async def flush(self):
        self.flushes += 1

    async def clear(self):
        self.cleared += 1


def _stream():
    s = DeepgramTTSStream(sample_rate=24000)
    s._conn = FakeConn()
    s._started = True
    s._flushed_event.set()
    return s


@pytest.mark.asyncio
async def test_late_first_confirmation_does_not_finish_the_reply():
    s = _stream()
    await s.send_text("First sentence.")
    await s.flush()                      # flush #1 (after the first sentence)
    await s.send_text("Second sentence.")
    await s.flush()                      # flush #2 (end of reply)
    assert not s._flushed_event.is_set()

    await s._on_flushed(None)            # confirmation of #1 arrives late
    assert not s._flushed_event.is_set(), "still waiting for the end-of-reply flush"
    await s._on_flushed(None)            # confirmation of #2
    assert s._flushed_event.is_set()


@pytest.mark.asyncio
async def test_single_flush_still_completes():
    s = _stream()
    await s.speak("Hello there.")
    assert not s._flushed_event.is_set()
    await s._on_flushed(None)
    assert s._flushed_event.is_set()


@pytest.mark.asyncio
async def test_flush_with_nothing_queued_is_skipped_so_no_wait_hangs():
    s = _stream()
    await s.speak("Only sentence.")      # flushes once
    await s.flush()                      # voice loop's end-of-reply flush with nothing new
    assert s._conn.flushes == 1
    await s._on_flushed(None)
    assert s._flushed_event.is_set()


@pytest.mark.asyncio
async def test_cancel_resets_the_count():
    s = _stream()
    await s.send_text("a")
    await s.flush()
    await s.send_text("b")
    await s.flush()
    await s.cancel()
    assert s._flushed_event.is_set() and s._pending_flushes == 0
    await s._on_flushed(None)            # stale confirmations after a barge-in are harmless
    assert s._pending_flushes == 0 and s._flushed_event.is_set()
