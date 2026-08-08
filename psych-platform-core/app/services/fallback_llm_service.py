import asyncio
import re
import time
import logging
from typing import Dict, Any, List

from app.services.llm_interface import LLMService

logger = logging.getLogger(__name__)

_RATE_LIMIT_BACKOFF = 65  # fallback when the provider gives no retry hint at all
# Ceiling on a provider-specified retry-after — a single spike shouldn't be
# able to bench a provider for an unreasonable stretch even if it asks to.
_MAX_RATE_LIMIT_BACKOFF = 300  # 5 minutes

_RETRY_AFTER_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _extract_retry_after(exc: Exception) -> float | None:
    """Best-effort extraction of a provider's own requested cooldown.

    Groq/OpenRouter (httpx-backed) expose a `retry-after` HTTP header on the
    response object attached to the exception. Gemini's google-genai SDK
    doesn't use that header — it embeds a `retryDelay` field in the JSON
    error body instead (e.g. "...'retryDelay': '11s'"), which only survives
    into str(exc), not as a structured attribute — hence the regex fallback.

    Why this matters: observed live that ignoring this and using a flat 65s
    backoff caused repeated 429s against a provider that was still well over
    budget — and each successive violation's own retry-after grew (502s ->
    695s in one Groq session), suggesting the blind retries were themselves
    making the outage longer, not just wasting a request each time.

    OpenRouter's *daily* quota exhaustion doesn't come with a retry-after at
    all — it reports `x-ratelimit-remaining: 0` plus an epoch-ms
    `x-ratelimit-reset`. Observed live (2026-08-05) with a 50/day free tier
    fully spent. That's a much stronger "don't bother retrying soon" signal
    than a missing header should be read as "use the naive 65s default" —
    checked explicitly below so a daily-exhausted provider doesn't get
    hit every 65s for the rest of the day.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        val = headers.get("retry-after")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        if headers.get("x-ratelimit-remaining") == "0":
            reset_ms = headers.get("x-ratelimit-reset")
            if reset_ms:
                try:
                    seconds_until_reset = (float(reset_ms) / 1000.0) - time.time()
                    if seconds_until_reset > 0:
                        return seconds_until_reset
                except ValueError:
                    pass
            # Remaining=0 with no usable reset time — still a much stronger
            # signal than "no info at all", so use the ceiling rather than
            # the short default.
            return _MAX_RATE_LIMIT_BACKOFF
    # Gemini's free tier reports a short retryDelay (e.g. "11s") even when
    # the actual violation is a PerDay quota — that delay is meaningless for
    # a cap that won't refresh for hours, and trusting it just means the next
    # attempt 19s later 429s again. "PerDay" in the quotaId is the signal
    # that no short wait will help.
    exc_str = str(exc)
    if "PerDay" in exc_str:
        return _MAX_RATE_LIMIT_BACKOFF
    match = _RETRY_AFTER_RE.search(exc_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None

# Per-provider budget for streaming calls specifically. Without this, a
# hung/rate-limited FIRST provider can consume a caller's entire outer
# timeout (see voice.py's LLM_FIRST_TOKEN_TIMEOUT_S) before this class ever
# gets a chance to move on to the next tier — observed live: OpenRouter
# hanging for 8s+ ate the whole budget, so Gemini (which may have been fine)
# never got tried. This gives each provider its own bounded window so a
# 3-tier chain can actually fail over within a reasonable total.
_STREAM_STEP_TIMEOUT_S = 6.0


class FallbackLLMService(LLMService):
    """
    Transparent waterfall: tries each service in order, returns first success.
    Circuit-breaker: once a service returns a 429 / rate-limit error it is
    skipped for _RATE_LIMIT_BACKOFF seconds so subsequent calls go straight
    to the next provider with zero wasted round-trips.
    """

    def __init__(self, *services: LLMService):
        self._services = [s for s in services if s is not None]
        self._backoff_until: dict[int, float] = {}

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
            "FallbackLLM: %s rate-limited — skipping for %.0fs (%s)",
            type(svc).__name__, backoff, source,
        )

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        s = str(exc)
        return "429" in s or "rate limit" in s.lower() or "too many requests" in s.lower()

    async def _try_all(self, method_name: str, *args, **kwargs):
        last_exc: Exception | None = None
        for svc in self._services:
            if not self._available(svc):
                logger.debug("FallbackLLM: %s skipped (backoff active)", type(svc).__name__)
                continue
            method = getattr(svc, method_name, None)
            if method is None:
                continue
            try:
                # Bounded per-provider — without this, a hung/rate-limited
                # first provider can consume a caller's entire outer timeout
                # before this class ever gets to try the next tier.
                return await asyncio.wait_for(method(*args, **kwargs), timeout=_STREAM_STEP_TIMEOUT_S)
            except Exception as exc:
                if self._is_rate_limit(exc) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._trip(svc, exc)
                else:
                    logger.warning(
                        "FallbackLLM: %s.%s failed (%s) — trying next",
                        type(svc).__name__, method_name, exc,
                    )
                last_exc = exc
        raise last_exc or RuntimeError(f"All services unavailable for {method_name}")

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        return await self._try_all("analyze_sentiment", text)

    async def is_utterance_complete(self, text: str) -> bool:
        """Only GroqService implements this today; other services are
        skipped via _try_all's getattr(..., None) guard. Exposed here so
        voice.py's hasattr(sentiment_service, "is_utterance_complete") check
        actually finds it — previously sentiment_service was a
        FallbackLLMService with no such attribute at all, so that hasattr
        check always failed and the semantic turn-completion fast path
        (see voice.py's pump_stt_events) was silently dead code."""
        try:
            return await self._try_all("is_utterance_complete", text)
        except Exception:
            return False

    async def analyze_sentiment_raw(self, prompt: str) -> Dict[str, Any]:
        return await self._try_all("analyze_sentiment_raw", prompt)

    async def generate_therapeutic_response(
        self, history: List[Dict], context: str, mood: str
    ) -> str:
        return await self._try_all("generate_therapeutic_response", history, context, mood)

    async def generate_response_for_move(self, history, system_prompt: str) -> str:
        return await self._try_all("generate_response_for_move", history, system_prompt)

    async def stream_response_for_move(self, history, system_prompt: str):
        """Streaming sibling of generate_response_for_move. Only falls
        through to the next service if the CURRENT one fails before yielding
        anything — once tokens have already been sent downstream (e.g. to
        TTS), starting a second provider from scratch mid-stream would
        produce garbled, duplicated output, which is worse than just ending
        the turn where it is."""
        for svc in self._services:
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
                        "FallbackLLM: %s.stream_response_for_move failed mid-stream "
                        "after partial output — ending stream rather than risking "
                        "garbled duplicate content: %s",
                        type(svc).__name__, exc,
                    )
                    return
                if self._is_rate_limit(exc) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._trip(svc, exc)
                else:
                    logger.warning(
                        "FallbackLLM: %s.stream_response_for_move failed (%s) — trying next",
                        type(svc).__name__, exc,
                    )
        yield "I'm here with you."

    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        return await self._try_all("summarize_conversation", messages, mood)

    async def stream_therapeutic_response(
        self, history, mood: str, context: str, language: str = "en"
    ):
        """Used by /chat/stream, /chat/stream/sentences, and /ws/chat.
        Previously had NO per-provider timeout at all — unlike
        stream_response_for_move (hardened while fixing voice's silent-hang
        bug earlier this session), a hung/rate-limited provider here could
        stall a text-chat turn indefinitely with nothing reaching the
        client. Same fix applied: bound each provider's per-chunk wait, and
        stop (rather than retry a second provider) once any output has
        already been yielded, since restarting mid-stream on a new provider
        would produce garbled/duplicated text."""
        for svc in self._services:
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
                        "FallbackLLM: %s.stream_therapeutic_response failed mid-stream "
                        "after partial output — ending stream rather than risking "
                        "garbled duplicate content: %s",
                        type(svc).__name__, exc,
                    )
                    return
                if self._is_rate_limit(exc) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    self._trip(svc, exc)
                else:
                    logger.warning(
                        "FallbackLLM: %s.stream failed (%s) — trying next",
                        type(svc).__name__, exc,
                    )
        yield "I'm here with you."
