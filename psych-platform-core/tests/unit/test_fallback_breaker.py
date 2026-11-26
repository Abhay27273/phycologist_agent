"""
FallbackLLMService circuit-breaker behaviour.

Regression for 2026-11-25: three providers each exceeded the 6s first-token
budget within ~12s of each other; a timeout benched each one for the full 65s
rate-limit backoff, so for the next minute NO provider was attempted and every
reply was the canned "I'm here with you.".
"""
import asyncio
import time

import pytest

import app.services.fallback_llm_service as fl
from app.services.fallback_llm_service import FallbackLLMService


class FakeSvc:
    def __init__(self, name, tokens=("hello", " there"), exc=None):
        self.name, self.tokens, self.exc, self.calls = name, tokens, exc, 0

    async def stream_response_for_move(self, history, system_prompt):
        self.calls += 1
        if self.exc:
            raise self.exc
        for t in self.tokens:
            yield t

    async def analyze_sentiment_raw(self, prompt):
        self.calls += 1
        if self.exc:
            raise self.exc
        return {"mood": "calm", "risk_score": 0}


async def collect(svc):
    return "".join([t async for t in svc.stream_response_for_move([], "sys")])


def test_timeout_bench_is_short_not_the_rate_limit_default():
    a, b = FakeSvc("a"), FakeSvc("b")
    chain = FallbackLLMService(a, b)
    before = time.monotonic()
    chain._trip(a, asyncio.TimeoutError())
    bench = chain._backoff_until[id(a)] - before
    assert fl._TIMEOUT_BACKOFF - 1 <= bench <= fl._TIMEOUT_BACKOFF + 1
    assert bench < fl._RATE_LIMIT_BACKOFF


def test_a_real_429_still_gets_the_long_default_bench():
    a = FakeSvc("a")
    chain = FallbackLLMService(a)
    before = time.monotonic()
    chain._trip(a, RuntimeError("Error code: 429 - rate limit"))
    assert chain._backoff_until[id(a)] - before >= fl._RATE_LIMIT_BACKOFF - 1


@pytest.mark.asyncio
async def test_all_benched_with_a_short_cooldown_waits_instead_of_going_canned(monkeypatch):
    monkeypatch.setattr(fl, "_MAX_ALL_BENCHED_WAIT_S", 3.0)
    a, b = FakeSvc("a", tokens=("from-a",)), FakeSvc("b", tokens=("from-b",))
    chain = FallbackLLMService(a, b)
    now = time.monotonic()
    chain._backoff_until[id(a)] = now + 0.3   # soonest
    chain._backoff_until[id(b)] = now + 2.0
    t0 = time.monotonic()
    out = await collect(chain)
    assert out == "from-a", "should wait for the soonest provider and use it"
    assert 0.2 <= time.monotonic() - t0 < 2.0
    assert b.calls == 0


@pytest.mark.asyncio
async def test_all_benched_with_a_long_cooldown_is_respected_and_falls_back_canned():
    """Long cooldowns (provider-specified retry-after, daily quotas) must NOT be
    retried: blind retries into an over-budget provider were observed to
    lengthen its outage."""
    a, b = FakeSvc("a"), FakeSvc("b")
    chain = FallbackLLMService(a, b)
    now = time.monotonic()
    chain._backoff_until[id(a)] = now + 120
    chain._backoff_until[id(b)] = now + 200
    out = await collect(chain)
    assert out == "I'm here with you."
    assert a.calls == 0 and b.calls == 0


@pytest.mark.asyncio
async def test_normal_path_unchanged_uses_first_available():
    a, b = FakeSvc("a", tokens=("A",)), FakeSvc("b", tokens=("B",))
    chain = FallbackLLMService(a, b)
    assert await collect(chain) == "A"
    # a benched -> b used immediately, no waiting
    chain._backoff_until[id(a)] = time.monotonic() + 60
    t0 = time.monotonic()
    assert await collect(chain) == "B"
    assert time.monotonic() - t0 < 0.5


@pytest.mark.asyncio
async def test_try_all_also_waits_for_a_short_cooldown(monkeypatch):
    monkeypatch.setattr(fl, "_MAX_ALL_BENCHED_WAIT_S", 3.0)
    a = FakeSvc("a")
    chain = FallbackLLMService(a)
    chain._backoff_until[id(a)] = time.monotonic() + 0.3
    out = await chain.analyze_sentiment_raw("p")
    assert out["mood"] == "calm" and a.calls == 1


@pytest.mark.asyncio
async def test_try_all_with_long_cooldown_raises_as_before():
    a = FakeSvc("a")
    chain = FallbackLLMService(a)
    chain._backoff_until[id(a)] = time.monotonic() + 120
    with pytest.raises(RuntimeError):
        await chain.analyze_sentiment_raw("p")
    assert a.calls == 0
