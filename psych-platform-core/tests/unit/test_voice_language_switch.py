"""
Voice calls follow the speaker's language turn by turn (auto mode).
No network: speech engines are faked.
"""
import asyncio

import pytest

from app.api.routes import voice as voice_routes
from app.api.routes.voice import VoiceSession, _select_voice_provider
from app.core.config import settings
from app.services.routed_tts import LanguageRoutedTTS
from app.services.sarvam_voice_service import SarvamSTTStream, SarvamTTSStream
from app.services.voice_service import DeepgramSTTStream


class FakeTTS:
    def __init__(self, name, start_ok=True):
        self.name = name
        self.start_ok = start_ok
        self.spoken = []
        self.cancelled = 0
        self.closed = False
        self._q = asyncio.Queue()

    async def start(self):
        return self.start_ok

    async def send_text(self, t):
        self.spoken.append(t)

    async def flush(self):
        pass

    async def speak(self, t):
        self.spoken.append(t)
        await self._q.put(f"{self.name}:{t}".encode())

    async def wait_until_flushed(self, timeout=10.0):
        pass

    async def cancel(self):
        self.cancelled += 1

    async def audio_chunks(self):
        while True:
            yield await self._q.get()

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_each_reply_goes_to_the_engine_for_its_language():
    en, hi = FakeTTS("en"), FakeTTS("hi")
    r = LanguageRoutedTTS(english=en, indic=hi)
    assert await r.start()

    r.set_language("en")
    await r.speak("hello")
    r.set_language("hi")
    await r.speak("नमस्ते")
    r.set_language("hinglish")
    await r.speak("kaise ho")
    r.set_language("en")
    await r.speak("fine thanks")

    assert en.spoken == ["hello", "fine thanks"]
    assert hi.spoken == ["नमस्ते", "kaise ho"]
    await r.close()


@pytest.mark.asyncio
async def test_audio_from_both_engines_arrives_on_one_stream():
    en, hi = FakeTTS("en"), FakeTTS("hi")
    r = LanguageRoutedTTS(english=en, indic=hi)
    await r.start()
    got = []

    async def consume():
        async for chunk in r.audio_chunks():
            got.append(chunk)
            if len(got) == 3:
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)          # let the forwarders start before audio exists
    # In a real call one reply finishes before the next begins, so the engines
    # never produce audio in the same instant.
    r.set_language("en")
    await r.speak("a")
    await asyncio.sleep(0.02)
    r.set_language("hi")
    await r.speak("b")
    await asyncio.sleep(0.02)
    r.set_language("en")
    await r.speak("c")
    await asyncio.wait_for(task, timeout=2)
    assert got == [b"en:a", b"hi:b", b"en:c"]
    await r.close()


@pytest.mark.asyncio
async def test_missing_or_failed_engine_is_covered_by_the_other():
    # Sarvam unavailable: Hindi replies still get spoken (English voice) rather than silence.
    en = FakeTTS("en")
    r = LanguageRoutedTTS(english=en, indic=FakeTTS("hi", start_ok=False))
    assert await r.start()
    r.set_language("hi")
    await r.speak("नमस्ते")
    assert en.spoken == ["नमस्ते"]
    await r.close()

    # Deepgram unavailable: English replies are spoken by the Indic engine.
    hi = FakeTTS("hi")
    r2 = LanguageRoutedTTS(english=FakeTTS("en", start_ok=False), indic=hi)
    assert await r2.start()
    r2.set_language("en")
    await r2.speak("hello")
    assert hi.spoken == ["hello"]
    await r2.close()

    r3 = LanguageRoutedTTS(english=FakeTTS("a", start_ok=False), indic=FakeTTS("b", start_ok=False))
    assert not await r3.start()


@pytest.mark.asyncio
async def test_barge_in_cancels_both_engines_and_drops_queued_audio():
    en, hi = FakeTTS("en"), FakeTTS("hi")
    r = LanguageRoutedTTS(english=en, indic=hi)
    await r.start()
    r._merged.put_nowait(b"stale")
    await r.cancel()
    assert en.cancelled == 1 and hi.cancelled == 1
    assert r._merged.empty()
    await r.close()


@pytest.mark.asyncio
async def test_a_failing_synthesis_request_does_not_end_the_audio_pump():
    class Flaky(FakeTTS):
        def __init__(self):
            super().__init__("hi")
            self.calls = 0

        async def audio_chunks(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Sarvam TTS error: HTTP 500")
            while True:
                yield await self._q.get()

    flaky = Flaky()
    r = LanguageRoutedTTS(english=FakeTTS("en"), indic=flaky)
    await r.start()
    got = []

    async def consume():
        async for chunk in r.audio_chunks():
            got.append(chunk)
            return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)        # first generator raised; forwarder must have resumed
    r.set_language("hi")
    await r.speak("बाद में")
    await asyncio.wait_for(task, timeout=2)
    assert got == [b"hi:\xe0\xa4\xac\xe0\xa4\xbe\xe0\xa4\xa6 \xe0\xa4\xae\xe0\xa5\x87\xe0\xa4\x82"]
    await r.close()


# ---- provider selection ------------------------------------------------------

def test_auto_with_sarvam_uses_autodetect_stt_and_routed_tts(monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "k")
    monkeypatch.setattr(settings, "DEEPGRAM_API_KEY", "k")
    stt, tts = _select_voice_provider("auto")
    assert isinstance(stt, SarvamSTTStream)
    assert "language-code=unknown" in stt._url()
    assert isinstance(tts, LanguageRoutedTTS)


def test_auto_without_sarvam_degrades_to_english_deepgram(monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
    monkeypatch.setattr(settings, "DEEPGRAM_API_KEY", "k")
    stt, tts = _select_voice_provider("auto")
    assert isinstance(stt, DeepgramSTTStream)
    assert not isinstance(tts, LanguageRoutedTTS)


def test_pinned_hindi_uses_sarvam_only(monkeypatch):
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "k")
    monkeypatch.setattr(settings, "DEEPGRAM_API_KEY", "k")
    stt, tts = _select_voice_provider("hi")
    assert isinstance(stt, SarvamSTTStream) and "language-code=hi-IN" in stt._url()
    assert isinstance(tts, SarvamTTSStream)


# ---- per-turn language -------------------------------------------------------

class _Bare(VoiceSession):
    def __init__(self, mode, current):
        self.session_id = "t"
        self.language_mode = mode
        self.session_language = current
        self.tts = FakeTTS("x")


def test_language_follows_the_speaker():
    s = _Bare("auto", "en")
    assert s._resolve_turn_language("मुझे नींद नहीं आती", "hi") == "hi"
    s.session_language = "hi"
    assert s._resolve_turn_language("I think I am okay now, thanks for listening", "en") == "en"


def test_filler_words_do_not_flip_the_voice_mid_hindi():
    s = _Bare("auto", "hi")
    for filler in ("ok", "yes", "hmm okay", "thank you"):
        assert s._resolve_turn_language(filler, "en") == "hi", filler
    s2 = _Bare("auto", "en")
    assert s2._resolve_turn_language("ok", "en") == "en"


def test_pinned_language_is_not_overridden_by_stickiness():
    s = _Bare("en", "en")
    assert s._resolve_turn_language("मुझे नींद नहीं आती", "hi") == "hi"  # detection untouched when pinned


def test_use_language_points_the_routed_tts():
    en, hi = FakeTTS("en"), FakeTTS("hi")
    r = LanguageRoutedTTS(english=en, indic=hi)
    r._started = [en, hi]
    s = _Bare("auto", "en")
    s.tts = r
    s._use_language("hinglish")
    assert s.session_language == "hinglish" and r.active_name == "indic"
    s._use_language("en")
    assert r.active_name == "english"


def test_voice_follows_the_script_of_the_reply_not_just_the_input():
    """English input but a Hindi reply (the model slipped after a Hindi conversation):
    the English voice must not read Devanagari."""
    en, hi = FakeTTS("en"), FakeTTS("hi")
    r = LanguageRoutedTTS(english=en, indic=hi)
    r._started = [en, hi]
    s = _Bare("auto", "en")
    s.tts = r
    s._use_language("en")
    s._align_voice_to_reply("आप इस डर के साथ अकेले नहीं हैं।", "en")
    assert r.active_name == "indic" and s.session_language == "hi"
    # a normal English reply to English input changes nothing
    s2 = _Bare("auto", "en")
    r2 = LanguageRoutedTTS(english=en, indic=hi)
    r2._started = [en, hi]
    s2.tts = r2
    s2._use_language("en")
    s2._align_voice_to_reply("That sounds exhausting.", "en")
    assert r2.active_name == "english"
    # pinned calls are never re-routed
    s3 = _Bare("en", "en")
    s3._align_voice_to_reply("आप अकेले नहीं हैं।", "en")
    assert s3.session_language == "en"
