import asyncio
import logging
import time
from typing import Dict, Any, List

from app.services.llm_interface import LLMService
from app.services.fallback_llm_service import (
    _extract_retry_after,
    _RATE_LIMIT_BACKOFF,
    _MAX_RATE_LIMIT_BACKOFF,
    _STREAM_STEP_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


class RoundRobinLLMService(LLMService):
    """
    Spreads calls across N same-provider instances (e.g. several Groq API
    keys) instead of always trying one first. Where FallbackLLMService is a
    priority waterfall (always try instance 0, only move to instance 1 on
    failure), this rotates the starting instance on every call so each
    key's own per-minute rate limit only sees ~1/N of total traffic.

    Built specifically because a single Groq key's ~12K-tokens/minute cap
    was the shared bottleneck throttling both text-chat and voice once
    OpenRouter/Gemini's daily free-tier quotas were exhausted (2026-08-06) —
    N keys round-robining multiplies the effective per-minute budget by N
    instead of everything funneling through one key's ceiling.

    Still falls through to the next key (in rotation order) on failure, so
    it degrades the same safe way FallbackLLMService does if a key is
    genuinely rate-limited or briefly down — this changes the ORDER
    calls are tried in per-call, not the safety behavior.
    """

    def __init__(self, *services: LLMService):
        self._services = [s for s in services if s is not None]
        if not self._services:
            raise ValueError("RoundRobinLLMService needs at least one service")
        self._backoff_until: dict[int, float] = {}
        self._next_index = 0

    def _available(self, svc: LLMService) -> bool:
        return time.monotonic() >= self._backoff_until.get(id(svc), 0)

    def _trip(self, svc: LLMService, exc: Exception | None = None) -> None:
        retry_after = _extract_retry_after(exc) if exc is not None else None
        if retry_after is not None:
            backoff = min(retry_after, _MAX_RATE_LIMIT_BACKOFF)
            source = "provider-specified"
        else:
            backoff = _RATE_LIMIT_BACKOFF
            source = "default"
        self._backoff_until[id(svc)] = time.monotonic() + backoff
        logger.warning(
            "RoundRobin: %s (key #%d) rate-limited — skipping for %.0fs (%s)",
            type(svc).__name__, self._services.index(svc), backoff, source,
        )

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        s = str(exc)
        return "429" in s or "rate limit" in s.lower() or "too many requests" in s.lower()

    def _rotation_order(self) -> list:
        """Each call starts from the next key in sequence — this is the
        actual load-balancing: without it, calling _services in a fixed
        order every time means later keys only ever get used once earlier
        ones are already rate-limited, which is fallback behavior, not
        spread-the-load behavior."""
        n = len(self._services)
        start = self._next_index
        self._next_index = (self._next_index + 1) % n
        return [self._services[(start + i) % n] for i in range(n)]

    async def _try_all(self, method_name: str, *args, **kwargs):
        last_exc: Exception | None = None
        for svc in self._rotation_order():
            if not self._available(svc):
                continue
            method = getattr(svc, method_name, None)
            if method is None:
                continue
            try:
                return await asyncio.wait_for(method(*args, **kwargs), timeout=_STREAM_STEP_TIMEOUT_S)
            except Exception as exc:
                if self._is_rate_limit(exc) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._trip(svc, exc)
                else:
                    logger.warning(
                        "RoundRobin: %s.%s failed (%s) — trying next key",
                        type(svc).__name__, method_name, exc,
                    )
                last_exc = exc
        raise last_exc or RuntimeError(f"All keys unavailable for {method_name}")

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        return await self._try_all("analyze_sentiment", text)

    async def analyze_sentiment_raw(self, prompt: str) -> Dict[str, Any]:
        return await self._try_all("analyze_sentiment_raw", prompt)

    async def is_utterance_complete(self, text: str) -> bool:
        try:
            return await self._try_all("is_utterance_complete", text)
        except Exception:
            return False

    async def generate_therapeutic_response(self, history: List[Dict], context: str, mood: str) -> str:
        return await self._try_all("generate_therapeutic_response", history, context, mood)

    async def generate_response_for_move(self, history, system_prompt: str) -> str:
        return await self._try_all("generate_response_for_move", history, system_prompt)

    async def stream_response_for_move(self, history, system_prompt: str):
        """Raises (doesn't yield a fallback string) when every key fails —
        this pool is meant to sit as one tier INSIDE an outer
        FallbackLLMService (e.g. Groq-pool -> OpenRouter -> Gemini). If this
        yielded its own "I'm here with you." on exhaustion, the outer layer
        would see that as a successful Groq response and never fall through
        to the next provider. Only the outermost layer in the whole chain
        should ever emit that generic fallback."""
        last_exc: Exception | None = None
        for svc in self._rotation_order():
            if not self._available(svc):
                continue
            method = getattr(svc, "stream_response_for_move", None)
            if method is None:
                continue
            yielded_any = False
            try:
                token_iter = method(history=history, system_prompt=system_prompt).__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(token_iter.__anext__(), timeout=_STREAM_STEP_TIMEOUT_S)
                    except StopAsyncIteration:
                        return
                    yielded_any = True
                    yield chunk
            except Exception as exc:
                if yielded_any:
                    logger.error(
                        "RoundRobin: %s.stream_response_for_move failed mid-stream "
                        "after partial output — ending stream: %s",
                        type(svc).__name__, exc,
                    )
                    return
                if self._is_rate_limit(exc) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._trip(svc, exc)
                else:
                    logger.warning(
                        "RoundRobin: %s.stream_response_for_move failed (%s) — trying next key",
                        type(svc).__name__, exc,
                    )
                last_exc = exc
        raise last_exc or RuntimeError("All Groq keys unavailable for stream_response_for_move")

    async def stream_therapeutic_response(self, history, mood: str, context: str, language: str = "en"):
        """Raises on exhaustion — see stream_response_for_move's docstring
        for why this pool must never emit its own fallback string."""
        last_exc: Exception | None = None
        for svc in self._rotation_order():
            if not self._available(svc):
                continue
            method = getattr(svc, "stream_therapeutic_response", None)
            if method is None:
                continue
            yielded_any = False
            try:
                token_iter = method(
                    history=history, mood=mood, context=context, language=language
                ).__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(token_iter.__anext__(), timeout=_STREAM_STEP_TIMEOUT_S)
                    except StopAsyncIteration:
                        return
                    yielded_any = True
                    yield chunk
            except Exception as exc:
                if yielded_any:
                    logger.error(
                        "RoundRobin: %s.stream_therapeutic_response failed mid-stream "
                        "after partial output — ending stream: %s",
                        type(svc).__name__, exc,
                    )
                    return
                if self._is_rate_limit(exc) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._trip(svc, exc)
                else:
                    logger.warning(
                        "RoundRobin: %s.stream failed (%s) — trying next key",
                        type(svc).__name__, exc,
                    )
                last_exc = exc
        raise last_exc or RuntimeError("All Groq keys unavailable for stream_therapeutic_response")

    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        return await self._try_all("summarize_conversation", messages, mood)
