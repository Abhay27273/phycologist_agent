"""
Sarvam voice wrappers: no network. The real protocol was verified against the
live API on 2026-11-26; these tests pin the parts that broke.
"""
import asyncio
import json

import pytest

from app.core.config import settings
from app.services import sarvam_voice_service as sv
from app.services.voice_service import (
    STT_ERROR,
    STT_SPEECH_STARTED,
    STT_TRANSCRIPT,
    TTS_AUDIO,
    TTS_FLUSHED,
)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "test-key")


def test_defaults_are_models_the_api_still_serves():
    # bulbul:v1 / v2 and the "meera" voice were retired; the API rejects them.
    assert settings.SARVAM_TTS_MODEL == "bulbul:v3"
    assert settings.SARVAM_TTS_VOICE != "meera"


def test_stt_url_is_the_streaming_endpoint_with_codemix():
    url = sv.SarvamSTTStream(sample_rate=16000, language_code="hi-IN")._url()
    assert url.startswith("wss://api.sarvam.ai/speech-to-text/ws?")
    assert "translate" not in url           # the old URL also translated to English
    assert "mode=codemix" in url
    assert "sample_rate=16000" in url
    assert "vad_signals=true" in url


class _FakeWS:
    def __init__(self, frames):
        self._frames = frames

    def __aiter__(self):
        async def gen():
            for f in self._frames:
                yield f
        return gen()


@pytest.mark.asyncio
async def test_stt_maps_sarvam_frames_to_events():
    stt = sv.SarvamSTTStream()
    stt._started = True
    frames = [
        json.dumps({"type": "events", "data": {"signal_type": "START_SPEECH"}}),
        json.dumps({"type": "events", "data": {"signal_type": "END_SPEECH"}}),
        json.dumps({"type": "data", "data": {"transcript": "  मुझे क्या करना चाहिए  "}}),
        json.dumps({"type": "data", "data": {"transcript": ""}}),          # empty: ignored
        "not json",                                                         # malformed: ignored
        json.dumps({"type": "error", "data": {"message": "quota"}}),
    ]
    await stt._receive_loop(_FakeWS(frames))
    got = []
    while not stt._queue.empty():
        got.append(stt._queue.get_nowait())
    kinds = [k for k, _ in got]
    assert kinds == [STT_SPEECH_STARTED, STT_TRANSCRIPT, STT_ERROR]
    ev = got[1][1]
    assert ev.transcript == "मुझे क्या करना चाहिए"
    assert ev.is_final and ev.speech_final   # one transcript per utterance, no interims


@pytest.mark.asyncio
async def test_tts_jobs_run_in_order_and_flush_only_after_the_last(monkeypatch):
    """The voice loop flushes twice per reply; audio must not interleave and
    wait_until_flushed must not release after the first job."""
    tts = sv.SarvamTTSStream()
    assert await tts.start()

    async def fake_synth(self, text):
        # first job is slower: concurrent execution would let job 2 overtake it
        await asyncio.sleep(0.15 if text == "one" else 0.01)
        await self._queue.put((TTS_AUDIO, text.encode()))

    monkeypatch.setattr(sv.SarvamTTSStream, "_synthesize", fake_synth)

    await tts.send_text("one")
    await tts.flush()
    await tts.send_text("two")
    await tts.flush()
    await asyncio.sleep(0.05)
    assert not tts._flushed_event.is_set(), "must stay 'speaking' while job 2 is still queued"
    await tts.wait_until_flushed(timeout=2)

    items = []
    while not tts._queue.empty():
        items.append(tts._queue.get_nowait())
    audio = [p for k, p in items if k == TTS_AUDIO]
    assert audio == [b"one", b"two"]
    assert [k for k, _ in items].count(TTS_FLUSHED) == 1
    assert items[-1][0] == TTS_FLUSHED
    await tts.close()


@pytest.mark.asyncio
async def test_tts_cancel_drops_queued_work(monkeypatch):
    tts = sv.SarvamTTSStream()
    await tts.start()

    async def slow(self, text):
        await asyncio.sleep(5)

    monkeypatch.setattr(sv.SarvamTTSStream, "_synthesize", slow)
    await tts.speak("one")
    await tts.send_text("two")
    await tts.flush()
    await asyncio.sleep(0.02)
    await tts.cancel()
    assert tts._flushed_event.is_set()
    assert tts._pending == 0 and tts._jobs.empty()
    await tts.close()
