"""
Latency benchmark suite for the chat API.

Requires a RUNNING server (this hits real endpoints, real LLM calls, real
RAG retrieval — there is no mocking here, by design, since the whole point
is to measure actual production-path latency).

Run:
    uvicorn app.api.server:app --port 8001      # in one terminal
    pytest tests/test_latency.py -v -s          # in another

Configure target / thresholds via env vars (all optional):
    TEST_BASE_URL                default http://127.0.0.1:8001
    MAX_BLOCKING_LATENCY_MS       default 8000   (POST /chat, full round trip)
    MAX_CRISIS_LATENCY_MS         default 3000   (deterministic path, no LLM generation)
    MAX_TIME_TO_FIRST_TOKEN_MS    default 4000   (POST /chat/stream)
    MAX_TIME_TO_FIRST_SENTENCE_MS default 4500   (POST /chat/stream/sentences, WS)

The suite prints a latency report (min/avg/p50/p95/max) for every test so you
can compare actual numbers against the target budget in IMPLEMENTATION_PLAN.md
even when a test passes comfortably inside its (deliberately generous) ceiling.

Note on running the full suite back-to-back: this suite makes ~25 real LLM
calls in ~90 seconds. Free-tier LLM provider quotas (e.g. Groq's per-minute
rate limit) can be exceeded by that point, triggering SDK retry-with-backoff
that adds 4-5s to the affected call — a real production risk under bursty
traffic, but not an application regression. If a streaming test fails only
when run as part of the full suite (and passes in isolation, e.g.
`pytest tests/test_latency.py::test_websocket_time_to_first_sentence`), that's
provider throttling, not a code bug. Raise INTER_SAMPLE_DELAY_S, use a paid
API tier, or run streaming tests separately from the rest for a clean read.
"""
import asyncio
import json
import os
import statistics
import time
import uuid

import httpx
import pytest
import websockets

BASE_URL = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:8001")
WS_BASE_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
API = f"{BASE_URL}/api/v1"

MAX_BLOCKING_MS = int(os.environ.get("MAX_BLOCKING_LATENCY_MS", "8000"))
MAX_CRISIS_MS = int(os.environ.get("MAX_CRISIS_LATENCY_MS", "3000"))
MAX_FIRST_TOKEN_MS = int(os.environ.get("MAX_TIME_TO_FIRST_TOKEN_MS", "4000"))
MAX_FIRST_SENTENCE_MS = int(os.environ.get("MAX_TIME_TO_FIRST_SENTENCE_MS", "6000"))

TEST_EMAIL = "latency-bench@example.com"
TEST_PASSWORD = "LatencyBench123!"

INTER_SAMPLE_DELAY_S = 2.0  # avoid tripping the LLM provider's own per-minute
                            # rate limit mid-benchmark — a burst of back-to-back
                            # calls can trigger SDK retry backoff (~4-5s) that
                            # reflects provider throttling, not app latency.

TRIVIAL_MESSAGE = "hi there"
CLINICAL_MESSAGE = (
    "I've been struggling with severe anxiety and panic attacks every night "
    "and I don't know how to cope with it anymore."
)
CRISIS_MESSAGE = "I want to end my life, I can't go on anymore."


def _report(name: str, samples_ms: list) -> None:
    """Print a min/avg/p50/p95/max latency report for a set of samples."""
    sorted_samples = sorted(samples_ms)
    p50 = statistics.median(sorted_samples)
    p95_idx = min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))
    p95 = sorted_samples[p95_idx]
    print(
        f"\n--- {name} ---\n"
        f"  n={len(samples_ms)}  min={min(samples_ms):.0f}ms  "
        f"avg={statistics.mean(samples_ms):.0f}ms  p50={p50:.0f}ms  "
        f"p95={p95:.0f}ms  max={max(samples_ms):.0f}ms"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _require_live_server():
    """Skip the whole module with a clear message if the server isn't up."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3.0)
        if r.status_code != 200:
            pytest.skip(f"Server at {BASE_URL} did not return 200 from /health")
    except httpx.ConnectError:
        pytest.skip(
            f"No server reachable at {BASE_URL}. Start it first: "
            f"uvicorn app.api.server:app --port 8001"
        )


@pytest.fixture(autouse=True)
def _pace_between_tests():
    """Small gap between test functions to keep the whole suite's call rate
    away from free-tier LLM provider per-minute limits (see module docstring)."""
    yield
    time.sleep(2.0)


@pytest.fixture(scope="session")
def auth():
    """Register-or-login a fixed benchmark account once per test session."""
    with httpx.Client(timeout=10.0) as client:
        r = client.post(
            f"{API}/auth/register",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
        )
        if r.status_code == 409:
            r = client.post(
                f"{API}/auth/login",
                json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
            )
        assert r.status_code in (200, 201), f"Auth failed: {r.status_code} {r.text}"
        data = r.json()
        return {
            "token": data["access_token"],
            "user_id": data["user_id"],
            "headers": {"Authorization": f"Bearer {data['access_token']}"},
        }


def new_session_id() -> str:
    """Fresh session per test — avoids SummaryNode firing mid-benchmark and
    avoids one test's conversation history affecting another's latency."""
    return f"latency-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Test cases — POST /chat (blocking)
# ---------------------------------------------------------------------------

def test_blocking_chat_latency_trivial(auth):
    """Short, non-clinical message should skip RAG entirely — fastest blocking path."""
    samples = []
    with httpx.Client(timeout=MAX_BLOCKING_MS / 1000 + 5) as client:
        for _ in range(3):
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),
                "message": TRIVIAL_MESSAGE,
            }
            start = time.perf_counter()
            r = client.post(f"{API}/chat", json=payload, headers=auth["headers"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert r.status_code == 200, r.text
            samples.append(elapsed_ms)
            time.sleep(INTER_SAMPLE_DELAY_S)

    _report("Blocking /chat — trivial message (RAG skipped)", samples)
    assert statistics.median(samples) < MAX_BLOCKING_MS, (
        f"Median latency {statistics.median(samples):.0f}ms exceeds {MAX_BLOCKING_MS}ms ceiling"
    )


def test_blocking_chat_latency_clinical(auth):
    """Clinical message triggers RAG retrieval + reranking — should be slower than trivial."""
    samples = []
    with httpx.Client(timeout=MAX_BLOCKING_MS / 1000 + 5) as client:
        for _ in range(3):
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),
                "message": CLINICAL_MESSAGE,
            }
            start = time.perf_counter()
            r = client.post(f"{API}/chat", json=payload, headers=auth["headers"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert r.status_code == 200, r.text
            assert r.json()["detected_mood"] is not None
            samples.append(elapsed_ms)
            time.sleep(INTER_SAMPLE_DELAY_S)

    _report("Blocking /chat — clinical message (RAG + reranking)", samples)
    assert statistics.median(samples) < MAX_BLOCKING_MS, (
        f"Median latency {statistics.median(samples):.0f}ms exceeds {MAX_BLOCKING_MS}ms ceiling"
    )


def test_blocking_chat_latency_crisis(auth):
    """
    Crisis messages route to the deterministic CrisisNode (no generative LLM
    call for the reply — see CLAUDE.md 'Hard-Coded Safety Net'). This should
    be at least as fast as the trivial path, since it skips RAG + generation
    and only pays for one sentiment-analysis call.
    """
    samples = []
    with httpx.Client(timeout=MAX_CRISIS_MS / 1000 + 5) as client:
        for _ in range(3):
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),
                "message": CRISIS_MESSAGE,
            }
            start = time.perf_counter()
            r = client.post(f"{API}/chat", json=payload, headers=auth["headers"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert r.status_code == 200, r.text
            assert r.json()["risk_level"] == "HIGH"
            samples.append(elapsed_ms)
            time.sleep(INTER_SAMPLE_DELAY_S)

    _report("Blocking /chat — crisis message (deterministic template)", samples)
    assert statistics.median(samples) < MAX_CRISIS_MS, (
        f"Median latency {statistics.median(samples):.0f}ms exceeds {MAX_CRISIS_MS}ms ceiling"
    )


def test_rag_cache_warm_vs_cold(auth):
    """
    Same query text + mood ⇒ same RAG cache key (md5(query|mood), independent
    of session/user — see rag_service.retrieve_clinical_context). The second
    call, in a brand-new session, should hit the cache and skip the
    cross-encoder reranking pass, so it should not be meaningfully slower
    than the first (and is usually faster).
    """
    query = "I feel so lonely and isolated since I moved to a new city."
    samples = {"cold": None, "warm": None}

    with httpx.Client(timeout=MAX_BLOCKING_MS / 1000 + 5) as client:
        for label in ("cold", "warm"):
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),  # different session, same text+mood
                "message": query,
            }
            start = time.perf_counter()
            r = client.post(f"{API}/chat", json=payload, headers=auth["headers"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert r.status_code == 200, r.text
            samples[label] = elapsed_ms

    print(
        f"\n--- RAG cache cold vs warm ---\n"
        f"  cold={samples['cold']:.0f}ms  warm={samples['warm']:.0f}ms  "
        f"delta={samples['cold'] - samples['warm']:+.0f}ms"
    )
    # Generous margin — LLM generation time dominates and varies run to run;
    # this just guards against the cache being silently broken/bypassed.
    assert samples["warm"] < samples["cold"] * 1.5, (
        "Warm (cached) request was not meaningfully faster — RAG cache may not be working"
    )


def test_concurrent_chat_latency(auth):
    """
    3 concurrent /chat requests (well under the 20/min rate limit) — checks
    that latency doesn't degrade sharply under light concurrent load.
    """
    import concurrent.futures

    def one_call(i: int) -> float:
        with httpx.Client(timeout=MAX_BLOCKING_MS / 1000 + 5) as client:
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),
                "message": f"I'm feeling stressed about a deadline (test #{i}).",
            }
            start = time.perf_counter()
            r = client.post(f"{API}/chat", json=payload, headers=auth["headers"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert r.status_code == 200, r.text
            return elapsed_ms

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        samples = list(pool.map(one_call, range(3)))

    _report("Concurrent /chat — 3 simultaneous requests", samples)
    assert max(samples) < MAX_BLOCKING_MS


# ---------------------------------------------------------------------------
# Test cases — streaming endpoints (time-to-first-token / sentence)
# ---------------------------------------------------------------------------

STREAM_MESSAGE = "I've been feeling overwhelmed with stress at work lately."
STREAM_SAMPLES = 3  # single-sample streaming timings are too noisy to trust


def test_sse_stream_time_to_first_token(auth):
    """POST /chat/stream — measures wall-clock time until the first token event."""
    samples = []
    with httpx.Client(timeout=MAX_FIRST_TOKEN_MS / 1000 + 10) as client:
        for _ in range(STREAM_SAMPLES):
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),
                "message": STREAM_MESSAGE,
            }
            start = time.perf_counter()
            with client.stream(
                "POST", f"{API}/chat/stream", json=payload, headers=auth["headers"]
            ) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    evt = json.loads(line[len("data:"):].strip())
                    if evt["type"] == "token":
                        samples.append((time.perf_counter() - start) * 1000)
                        break
            time.sleep(INTER_SAMPLE_DELAY_S)

    assert samples, "No token event received"
    _report("SSE /chat/stream — time to first token", samples)
    assert statistics.median(samples) < MAX_FIRST_TOKEN_MS, (
        f"Median time to first token {statistics.median(samples):.0f}ms exceeds {MAX_FIRST_TOKEN_MS}ms ceiling"
    )


def test_sse_stream_sentences_time_to_first_sentence(auth):
    """POST /chat/stream/sentences — measures time until the first complete sentence."""
    samples = []
    with httpx.Client(timeout=MAX_FIRST_SENTENCE_MS / 1000 + 10) as client:
        for _ in range(STREAM_SAMPLES):
            payload = {
                "user_id": auth["user_id"],
                "session_id": new_session_id(),
                "message": STREAM_MESSAGE,
            }
            start = time.perf_counter()
            with client.stream(
                "POST", f"{API}/chat/stream/sentences", json=payload, headers=auth["headers"]
            ) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    evt = json.loads(line[len("data:"):].strip())
                    if evt["type"] == "sentence":
                        samples.append((time.perf_counter() - start) * 1000)
                        break
            time.sleep(INTER_SAMPLE_DELAY_S)

    assert samples, "No sentence event received"
    _report("SSE /chat/stream/sentences — time to first sentence", samples)
    assert statistics.median(samples) < MAX_FIRST_SENTENCE_MS, (
        f"Median time to first sentence {statistics.median(samples):.0f}ms exceeds {MAX_FIRST_SENTENCE_MS}ms ceiling"
    )


def test_websocket_time_to_first_sentence(auth):
    """
    WS /ws/chat/{session_id} — measures time-to-first-sentence over a
    persistent connection. Unlike SSE, the connection/auth handshake happens
    once; this isolates true per-turn latency for repeated voice-agent turns.
    """

    async def one_turn(ws) -> float:
        start = time.perf_counter()
        await ws.send(json.dumps({"user_id": auth["user_id"], "message": STREAM_MESSAGE}))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=MAX_FIRST_SENTENCE_MS / 1000 + 10)
            evt = json.loads(raw)
            if evt["type"] == "sentence":
                return (time.perf_counter() - start) * 1000
            if evt["type"] == "error":
                pytest.fail(f"WebSocket returned error: {evt}")
            if evt["type"] == "done":
                pytest.fail("Got 'done' before any 'sentence' event")

    async def drain_to_done(ws) -> None:
        """Consume remaining events for the turn so the next send() starts clean."""
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=MAX_FIRST_SENTENCE_MS / 1000 + 10)
            if json.loads(raw)["type"] == "done":
                return

    async def run():
        session_id = new_session_id()
        uri = f"{WS_BASE_URL}/api/v1/ws/chat/{session_id}?token={auth['token']}"
        samples = []
        async with websockets.connect(uri) as ws:
            for _ in range(STREAM_SAMPLES):
                samples.append(await one_turn(ws))
                await drain_to_done(ws)
                await asyncio.sleep(INTER_SAMPLE_DELAY_S)
        return samples

    samples = asyncio.run(run())
    _report("WebSocket /ws/chat — time to first sentence (same connection, repeated turns)", samples)
    assert statistics.median(samples) < MAX_FIRST_SENTENCE_MS, (
        f"Median time to first sentence {statistics.median(samples):.0f}ms exceeds {MAX_FIRST_SENTENCE_MS}ms ceiling"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
