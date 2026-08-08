"""
One-time helper: synthesizes the speech fixtures tests/test_voice_latency.py
needs, using Deepgram's own streaming TTS (reuses app.services.voice_service,
which also exercises that wrapper as a side effect).

Needs DEEPGRAM_API_KEY set (same key the app itself uses). Run once; the
resulting .raw files are meant to be committed so the test suite doesn't
need to regenerate them.

Usage:
    python scripts/generate_voice_fixture.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.voice_service import DeepgramTTSStream  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

SAMPLES = {
    # Deliberately mild, unambiguous non-crisis distress — phrasing like
    # "I don't know how to cope with it anymore" scores right at the
    # risk_score >= 8 crisis threshold and accidentally routes this fixture
    # into the crisis path too, defeating the point of a RAG+LLM latency test.
    "voice_sample.raw": (
        "I've been feeling anxious about my job interview next week and "
        "could use some advice on managing the stress."
    ),
    "voice_crisis_sample.raw": "I want to end my life, I can't go on anymore.",
}


async def _synthesize(text: str) -> bytes:
    """Uses our own DeepgramTTSStream wrapper at 16kHz so the output is
    directly usable as STT input for the latency tests (Deepgram TTS can
    render at 16kHz just as well as the 24kHz we use for real playback)."""
    tts = DeepgramTTSStream(sample_rate=16000)
    if not await tts.start():
        raise RuntimeError("Could not start Deepgram TTS — check DEEPGRAM_API_KEY")

    chunks = []

    async def collect():
        async for chunk in tts.audio_chunks():
            chunks.append(chunk)

    collector = asyncio.create_task(collect())
    await tts.speak(text)
    await asyncio.sleep(3)  # give Deepgram time to finish generating
    collector.cancel()
    await tts.close()
    return b"".join(chunks)


async def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, text in SAMPLES.items():
        print(f"Synthesizing {filename}: {text!r}")
        audio = await _synthesize(text)
        out_path = FIXTURES_DIR / filename
        out_path.write_bytes(audio)
        print(f"  wrote {len(audio)} bytes -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
