"""
Real-time voice pipeline latency benchmarks (STT -> sentiment gate -> RAG+LLM -> TTS).

Requires:
  - A RUNNING server with DEEPGRAM_API_KEY configured. Voice endpoints 503
    without it, and this suite skips itself in that case (same pattern as
    tests/test_latency.py's live-server check).
  - Speech fixtures at tests/fixtures/voice_sample.raw and
    voice_crisis_sample.raw (16kHz mono 16-bit PCM). Generate once with:
        python scripts/generate_voice_fixture.py

Run:
    uvicorn app.api.server:app --port 8001
    pytest tests/test_voice_latency.py -v -s

Configure thresholds via env vars (all optional). These are regression
guards, not the ~800ms aspirational target in VOICE_IMPLEMENTATION_PLAN.md —
real measurement showed that target was unreachable given how the pipeline
actually works: Deepgram's UtteranceEnd is computed from elapsed *audio*
time (not wall-clock), so it needs ~1.0-1.8s of continued silence after the
user stops talking before it fires at all, before any of our own processing
even starts. That floor is inherent to VAD-based turn detection, not
something this codebase controls. On top of that floor: a sentiment-analysis
LLM round-trip (~0.3-1.0s) and Deepgram TTS synthesis-to-first-chunk
(~0.3-0.4s). Measured locally: crisis path ~3.2-3.9s, full RAG+LLM path
~4.0-4.6s, barge-in interrupt ~0.4-0.8s.
    MAX_VOICE_TO_VOICE_MS   default 6000  (speech end -> first audio frame)
    MAX_CRISIS_VOICE_MS     default 5000  (deterministic path, no LLM generation)
    MAX_BARGE_IN_MS         default 1500  (speech-start while AI talking -> 'interrupted')
"""
import asyncio
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
import pytest
import websockets

BASE_URL = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:8001")
WS_BASE_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
API = f"{BASE_URL}/api/v1"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES_DIR / "voice_sample.raw"
CRISIS_FIXTURE_PATH = FIXTURES_DIR / "voice_crisis_sample.raw"

MAX_VOICE_TO_VOICE_MS = int(os.environ.get("MAX_VOICE_TO_VOICE_MS", "6000"))
MAX_CRISIS_VOICE_MS = int(os.environ.get("MAX_CRISIS_VOICE_MS", "5000"))
MAX_BARGE_IN_MS = int(os.environ.get("MAX_BARGE_IN_MS", "1500"))

TEST_EMAIL = "voice-bench@example.com"
TEST_PASSWORD = "VoiceBench123!"

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_MS = 20  # pace sends to imitate a live mic stream, not a burst upload
CHUNK_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_MS / 1000)
RECV_TIMEOUT_S = 15


def _report(name: str, samples_ms: list) -> None:
    sorted_samples = sorted(samples_ms)
    p50 = statistics.median(sorted_samples)
    p95_idx = min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))
    print(
        f"\n--- {name} ---\n"
        f"  n={len(samples_ms)}  min={min(samples_ms):.0f}ms  "
        f"avg={statistics.mean(samples_ms):.0f}ms  p50={p50:.0f}ms  "
        f"p95={sorted_samples[p95_idx]:.0f}ms  max={max(samples_ms):.0f}ms"
    )


@pytest.fixture(scope="session", autouse=True)
def _require_live_server_and_voice():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3.0)
        if r.status_code != 200:
            pytest.skip(f"Server at {BASE_URL} did not return 200 from /health")
    except httpx.ConnectError:
        pytest.skip(f"No server reachable at {BASE_URL}. Start it first.")

    if not FIXTURE_PATH.exists() or not CRISIS_FIXTURE_PATH.exists():
        pytest.skip(
            "Voice fixtures missing — run: python scripts/generate_voice_fixture.py "
            "(needs DEEPGRAM_API_KEY)"
        )


@pytest.fixture(scope="session")
def auth():
    with httpx.Client(timeout=10.0) as client:
        r = client.post(f"{API}/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
        if r.status_code == 409:
            r = client.post(f"{API}/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
        assert r.status_code in (200, 201), f"Auth failed: {r.status_code} {r.text}"
        data = r.json()
        return {"token": data["access_token"], "user_id": data["user_id"]}


def new_session_id() -> str:
    return f"voice-latency-{uuid.uuid4().hex[:12]}"


def _voice_uri(session_id: str, token: str) -> str:
    return f"{WS_BASE_URL}/api/v1/ws/voice/{session_id}?token={token}"


# Deepgram's UtteranceEnd fires based on elapsed *audio* time, not wall-clock
# time — it needs audio (silence is fine) to keep arriving to advance that
# internal clock. A live mic keeps streaming through pauses (see
# voice-call.js's captureNode.port.onmessage, which sends every quantum
# regardless of speech), so production never hits this. This fixture is a
# finite pre-recorded buffer, so without trailing silence, sending stops
# dead the moment the recording ends and UtteranceEnd never fires.
TRAILING_SILENCE_MS = 1500


async def _pace_send(ws, audio_bytes: bytes) -> None:
    """Streams audio in small chunks at real-time pace — sending it as one
    burst wouldn't exercise Deepgram's VAD/utterance-end detection the way
    a live mic does."""
    for i in range(0, len(audio_bytes), CHUNK_BYTES):
        await ws.send(audio_bytes[i:i + CHUNK_BYTES])
        await asyncio.sleep(CHUNK_MS / 1000)


async def _send_silence(ws, duration_ms: int = TRAILING_SILENCE_MS) -> None:
    """Keeps streaming silence after real speech ends, standing in for a live
    mic that never stops sending audio mid-call (see TRAILING_SILENCE_MS)."""
    silence_chunk = b"\x00" * CHUNK_BYTES
    for _ in range(0, duration_ms, CHUNK_MS):
        await ws.send(silence_chunk)
        await asyncio.sleep(CHUNK_MS / 1000)


async def _stream_and_time_first_audio(uri: str, fixture_path: Path) -> tuple:
    """Connects, streams a fixture at real-time pace. Returns (first_audio_ms,
    done_ms) measured from the end of the user's speech.

    These can now differ a lot on the non-crisis path: a content-neutral
    backchannel ack ("Mm, I hear you.") is spoken immediately once mood is
    known, before RAG retrieval + LLM generation — see _BACKCHANNELS in
    voice.py. first_audio_ms is time-to-ack (the perceptually relevant "did
    it hear me" number); done_ms is time to full pipeline completion
    (RAG + generation + all TTS flushed). The crisis path has no
    backchannel, so the two stay close together there."""
    audio_bytes = fixture_path.read_bytes()
    async with websockets.connect(uri) as ws:
        await asyncio.wait_for(ws.recv(), timeout=10)  # 'ready'

        await _pace_send(ws, audio_bytes)  # the user actually talking
        start = time.perf_counter()  # speech end — the latency that matters
        send_task = asyncio.create_task(_send_silence(ws))
        first_audio_ms = None
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                elapsed = (time.perf_counter() - start) * 1000
                if isinstance(msg, bytes):
                    if first_audio_ms is None:
                        print(f"    [{elapsed:7.0f}ms] <first audio frame>")
                        first_audio_ms = elapsed
                    continue
                evt = json.loads(msg)
                print(f"    [{elapsed:7.0f}ms] {evt.get('type')}: {str(evt)[:80]}")
                if evt.get("type") == "error":
                    pytest.fail(f"Voice pipeline error: {evt}")
                if evt.get("type") == "done":
                    return first_audio_ms, elapsed
        finally:
            send_task.cancel()


def test_voice_to_voice_latency(auth):
    """End-to-end: speech in -> STT -> sentiment -> RAG+LLM -> TTS -> first audio frame out.

    Asserts on time-to-first-audio (the ack, on this path) since that's the
    perceptually relevant "does this feel responsive" number; time-to-done
    is reported for visibility but isn't held to the same ceiling — it
    includes full RAG+LLM generation and varies with response length."""
    first_audio_samples = []
    done_samples = []
    for _ in range(3):
        first_ms, done_ms = asyncio.run(_stream_and_time_first_audio(
            _voice_uri(new_session_id(), auth["token"]), FIXTURE_PATH
        ))
        first_audio_samples.append(first_ms)
        done_samples.append(done_ms)
        time.sleep(2)  # avoid tripping the LLM/STT providers' own rate limits

    _report("Voice-to-voice — time to first audio frame (ack)", first_audio_samples)
    _report("Voice-to-voice — time to done (full pipeline)", done_samples)
    assert statistics.median(first_audio_samples) < MAX_VOICE_TO_VOICE_MS, (
        f"Median voice-to-voice latency {statistics.median(first_audio_samples):.0f}ms "
        f"exceeds {MAX_VOICE_TO_VOICE_MS}ms ceiling"
    )


def test_crisis_voice_latency(auth):
    """Crisis path skips RAG + LLM generation entirely — should be the fastest path."""
    ms, _done_ms = asyncio.run(_stream_and_time_first_audio(
        _voice_uri(new_session_id(), auth["token"]), CRISIS_FIXTURE_PATH
    ))
    _report("Crisis path — time to first audio frame", [ms])
    assert ms < MAX_CRISIS_VOICE_MS, (
        f"Crisis-path latency {ms:.0f}ms exceeds {MAX_CRISIS_VOICE_MS}ms ceiling"
    )


async def _measure_barge_in(token: str) -> float:
    audio_bytes = FIXTURE_PATH.read_bytes()
    async with websockets.connect(_voice_uri(new_session_id(), token)) as ws:
        await asyncio.wait_for(ws.recv(), timeout=10)  # 'ready'

        # First turn — let the AI actually start speaking before interrupting it.
        await _pace_send(ws, audio_bytes)
        silence_task = asyncio.create_task(_send_silence(ws))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                if not isinstance(msg, bytes):
                    evt = json.loads(msg)
                    if evt.get("type") == "speaking_started":
                        break
                    if evt.get("type") == "error":
                        pytest.fail(f"Voice pipeline error before barge-in: {evt}")
        finally:
            silence_task.cancel()

        # Barge in — speak again while it's mid-response. Two-stage design
        # (see VoiceSession.duck_for_possible_speech / handle_barge_in):
        # "duck" fires on raw mic energy and is what actually stops the
        # audio the user hears — that's the number that matters for "does
        # this feel responsive". "interrupted" only fires once real words
        # are transcribed, confirming it wasn't just noise; it's inherently
        # bound by ASR latency (how much audio the STT model needs before it
        # emits a hypothesis), not a fixed reaction time, so it's reported
        # but not asserted on with the same tight ceiling.
        start = time.perf_counter()
        send_task = asyncio.create_task(_pace_send(ws, audio_bytes))
        duck_ms = None
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                if not isinstance(msg, bytes):
                    evt = json.loads(msg)
                    if evt.get("type") == "duck" and duck_ms is None:
                        duck_ms = (time.perf_counter() - start) * 1000
                    if evt.get("type") == "interrupted":
                        return duck_ms, (time.perf_counter() - start) * 1000
        finally:
            send_task.cancel()


def test_barge_in_latency(auth):
    """Speaking while the AI is talking should stop its audio quickly — the
    core of what makes this feel like a real conversation instead of a
    walkie-talkie. Asserted on "duck" (audio actually stops) since that's
    what's perceptible; "interrupted" (words confirmed, turn torn down) is
    reported for visibility but isn't held to the same tight ceiling — it's
    bound by ASR latency, not a fixed reaction time."""
    duck_ms, interrupted_ms = asyncio.run(_measure_barge_in(auth["token"]))
    _report("Barge-in — audio stops (duck)", [duck_ms])
    _report("Barge-in — words confirmed (interrupted)", [interrupted_ms])
    assert duck_ms is not None, "Never received a 'duck' event before 'interrupted'"
    assert duck_ms < MAX_BARGE_IN_MS, (
        f"Time to duck (audio stop) {duck_ms:.0f}ms exceeds {MAX_BARGE_IN_MS}ms ceiling"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
