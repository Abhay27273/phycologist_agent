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

Configure thresholds via env vars (all optional, deliberately looser than
the ~800ms target in VOICE_IMPLEMENTATION_PLAN.md — a 2-vCPU VM sharing
cores with the reranker won't hit the target consistently, this just
guards against real regressions):
    MAX_VOICE_TO_VOICE_MS   default 1200  (speech end -> first audio frame)
    MAX_CRISIS_VOICE_MS     default 800   (deterministic path, no LLM generation)
    MAX_BARGE_IN_MS         default 500   (speech-start while AI talking -> 'interrupted')
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

MAX_VOICE_TO_VOICE_MS = int(os.environ.get("MAX_VOICE_TO_VOICE_MS", "1200"))
MAX_CRISIS_VOICE_MS = int(os.environ.get("MAX_CRISIS_VOICE_MS", "800"))
MAX_BARGE_IN_MS = int(os.environ.get("MAX_BARGE_IN_MS", "500"))

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


async def _pace_send(ws, audio_bytes: bytes) -> None:
    """Streams audio in small chunks at real-time pace — sending it as one
    burst wouldn't exercise Deepgram's VAD/utterance-end detection the way
    a live mic does."""
    for i in range(0, len(audio_bytes), CHUNK_BYTES):
        await ws.send(audio_bytes[i:i + CHUNK_BYTES])
        await asyncio.sleep(CHUNK_MS / 1000)


async def _stream_and_time_first_audio(uri: str, fixture_path: Path) -> float:
    """Connects, streams a fixture at real-time pace, returns ms from the
    start of streaming to the first binary (audio) frame received."""
    audio_bytes = fixture_path.read_bytes()
    async with websockets.connect(uri) as ws:
        await asyncio.wait_for(ws.recv(), timeout=10)  # 'ready'

        start = time.perf_counter()
        send_task = asyncio.create_task(_pace_send(ws, audio_bytes))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                if isinstance(msg, bytes):
                    return (time.perf_counter() - start) * 1000
                evt = json.loads(msg)
                if evt.get("type") == "error":
                    pytest.fail(f"Voice pipeline error: {evt}")
        finally:
            send_task.cancel()


def test_voice_to_voice_latency(auth):
    """End-to-end: speech in -> STT -> sentiment -> RAG+LLM -> TTS -> first audio frame out."""
    samples = []
    for _ in range(3):
        ms = asyncio.run(_stream_and_time_first_audio(
            _voice_uri(new_session_id(), auth["token"]), FIXTURE_PATH
        ))
        samples.append(ms)
        time.sleep(2)  # avoid tripping the LLM/STT providers' own rate limits

    _report("Voice-to-voice — time to first audio frame", samples)
    assert statistics.median(samples) < MAX_VOICE_TO_VOICE_MS, (
        f"Median voice-to-voice latency {statistics.median(samples):.0f}ms "
        f"exceeds {MAX_VOICE_TO_VOICE_MS}ms ceiling"
    )


def test_crisis_voice_latency(auth):
    """Crisis path skips RAG + LLM generation entirely — should be the fastest path."""
    ms = asyncio.run(_stream_and_time_first_audio(
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
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
            if not isinstance(msg, bytes):
                evt = json.loads(msg)
                if evt.get("type") == "speaking_started":
                    break
                if evt.get("type") == "error":
                    pytest.fail(f"Voice pipeline error before barge-in: {evt}")

        # Barge in — speak again while it's mid-response.
        start = time.perf_counter()
        send_task = asyncio.create_task(_pace_send(ws, audio_bytes))
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
                if not isinstance(msg, bytes):
                    evt = json.loads(msg)
                    if evt.get("type") == "interrupted":
                        return (time.perf_counter() - start) * 1000
        finally:
            send_task.cancel()


def test_barge_in_latency(auth):
    """Speaking while the AI is talking should stop its audio quickly — the
    core of what makes this feel like a real conversation instead of a
    walkie-talkie."""
    ms = asyncio.run(_measure_barge_in(auth["token"]))
    _report("Barge-in — interrupt latency", [ms])
    assert ms < MAX_BARGE_IN_MS, (
        f"Barge-in latency {ms:.0f}ms exceeds {MAX_BARGE_IN_MS}ms ceiling"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
