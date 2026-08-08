"""
WS /api/v1/ws/voice/{session_id} — real-time speech-to-speech voice calls.

Cascaded architecture (STT -> sentiment/risk gate -> RAG+LLM -> TTS), NOT
end-to-end speech-to-speech — deliberately, so the crisis safety net (see
CrisisNode / CRISIS_MESSAGE) can inspect text and force a deterministic
response before anything is spoken. See VOICE_IMPLEMENTATION_PLAN.md for
the full architecture writeup and provider rationale.

Turn-taking is a hybrid detector (see pump_stt_events): Deepgram's acoustic
UtteranceEnd (~1000ms+ silence) is the always-fires fallback. Deepgram's
faster speech_final (~300ms) gates two independent, parallel fast-path
checks — a text-based LLM semantic check (GroqService.is_utterance_complete)
and an audio-based local model (turn_detector.SmartTurnDetector, ~12-65ms,
no network round-trip) — either of which can trigger a turn early once
confident. None of these can make turn-taking slower or less safe than
acoustic-only detection; a stall watchdog guarantees a reply even if all
of them fail to fire.

Client protocol:
  Binary frames  -> raw 16-bit PCM mono audio @ 16kHz (mic input)
  Text frames    -> reserved for future control messages (unused in V1)

Server protocol:
  Binary frames  -> raw 16-bit PCM mono audio @ 24kHz (TTS output)
  Text frames    -> JSON events:
    {"type": "ready"}
    {"type": "partial_transcript", "text": "..."}
    {"type": "final_transcript",   "text": "..."}
    {"type": "meta", "mood": "...", "risk_score": N}
    {"type": "speaking_started"}
    {"type": "speaking_ended"}
    {"type": "interrupted"}           -- barge-in: client must flush playback
    {"type": "done", "risk_level": "..."}
    {"type": "error", "message": "..."}
"""
import asyncio
import json
import logging
import random
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database import get_db
from app.infrastructure.models import ChatMessage
from app.graph.workflow import sentiment_service, therapy_llm_service, rag_service, gemini_service
from app.services.memory_service import extract_session_insights, MemoryService
from app.graph.nodes.strategy import StrategyNode
from app.graph.nodes.therapy import _needs_rag, _affect_valence_for_mood, _register_for_language
from app.services.therapeutic_prompt import build_therapeutic_system_prompt
from app.services.voice_service import (
    DeepgramSTTStream,
    DeepgramTTSStream,
    voice_enabled,
    STT_TRANSCRIPT,
    STT_SPEECH_STARTED,
    STT_UTTERANCE_END,
    STT_ERROR,
)
from app.services.sarvam_voice_service import (
    SarvamSTTStream,
    SarvamTTSStream,
    sarvam_enabled,
)
from app.services.voice_interface import STTStream, TTSStream
from app.services.turn_detector import smart_turn_detector
from app.api.routes.chat import (
    _get_or_create_user_session,
    _load_history,
    _load_longitudinal_context,
    _risk_level,
    _flush_sentences,
)
from app.graph.nodes.sentiment import SentimentNode, _detect_language
from app.graph.nodes.crisis import crisis_message_for

# Same calibrated sentiment path SentimentNode gives text-chat (passive-SI
# category, Hindi active-vs-passive distinction, somatic/tension handling,
# cognitive_distortion detection) — voice previously called the bare
# sentiment_service.analyze_sentiment(), which has none of that calibration.
# Mirrors workflow.py's own SentimentNode construction (same fallback chain).
_sentiment_node = SentimentNode(sentiment_service, fallback_llm_service=gemini_service)
_strategy_node = StrategyNode()

router = APIRouter()
logger = logging.getLogger(__name__)

# How long we'll sit on transcribed-but-unanswered speech before responding
# anyway. Comfortably above utterance_end_ms (1000ms) so it only fires when
# the normal triggers genuinely failed, not as part of routine turn-taking.
TURN_STALL_TIMEOUT_S = 2.5

# How long playback stays ducked waiting for mic energy to be confirmed as
# actual transcribed speech before we assume it was noise and resume.
BARGE_IN_CONFIRM_S = 1.2

# Rolling audio buffer for Smart Turn v3 — comfortably above its 8s analysis
# window (16-bit mono @ 16kHz = 32,000 bytes/sec).
_AUDIO_BUFFER_MAX_BYTES = 10 * 16000 * 2

# Fast-fail budgets for the two LLM touchpoints in a turn (sentiment analysis,
# response generation). Without these, a rate-limited provider can hang the
# whole turn for 30-60s+ with NOTHING sent to the client — the underlying
# SDKs retry internally on 429s with their own backoff, and the fallback
# chain (Groq -> OpenRouter -> Gemini) compounds that across multiple
# providers before our own except block ever gets control back. Confirmed
# live: this reproduced as a bare TimeoutError with zero events reaching the
# client, indistinguishable from the AI going silent mid-call. These timeouts
# bound that failure mode to a few seconds and let the existing error/fallback
# path actually fire instead of hanging silently.
#
# These are OUTER ceilings around the whole fallback chain, which itself gives
# each individual provider its own bounded window (see
# FallbackLLMService._STREAM_STEP_TIMEOUT_S = 6s). Confirmed live this outer
# ceiling must exceed the worst case of trying every tier at its own budget
# (up to 3 providers x 6s = 18s) — an 8s outer ceiling was cutting the chain
# off after the FIRST provider failed over, before a working second/third
# tier ever got a real chance. NONE of this affects normal-case latency — a
# healthy provider responds in 1-3s regardless of where this ceiling sits;
# it only bounds how long a genuinely degraded run of providers takes to
# recover, so raising it only matters for that already-broken case.
LLM_SENTIMENT_TIMEOUT_S = 25.0
LLM_FIRST_TOKEN_TIMEOUT_S = 25.0
LLM_STREAM_TOTAL_TIMEOUT_S = 35.0

# Backchannel acknowledgments spoken immediately once mood is known, before
# RAG retrieval + LLM generation (which the RAG path can add 1-3s to) — a
# standard technique for masking reply latency in voice agents: silence
# reads as "did it hear me?", a quick ack reads as "it's thinking". Content-
# neutral (no assumptions about what was said) so it's safe to fire this
# early. Not used on the crisis path — a generic filler ahead of the safety
# message would be actively inappropriate there. Text-level prosody tricks
# (ellipses, extra commas) were tested and showed no meaningful pacing
# effect on Aura-2 streaming — this list is deliberately plain text, not
# leaning on punctuation to "sound" more natural.
_BACKCHANNELS = [
    "Mm, I hear you.",
    "Okay, thank you for sharing that.",
    "I'm listening.",
    "I hear you.",
]

# Hinglish backchannels — used when session_language is hi/hinglish
_BACKCHANNELS_HI = [
    "Hmm, suno.",
    "Haan, batao.",
    "Main sun raha hoon.",
    "Achha, samjha.",
]


def _select_voice_provider(session_language: str) -> tuple[STTStream, TTSStream]:
    """
    Route to Sarvam for Hindi/Hinglish sessions; fall back to Deepgram for English
    or when Sarvam is not configured.
    The cascaded safety architecture (text checkpoint) is provider-independent.
    """
    if session_language in ("hi", "hinglish") and sarvam_enabled():
        return SarvamSTTStream(sample_rate=16000), SarvamTTSStream(sample_rate=24000)
    # Deepgram defaults to English-only decoding with no language hint, which
    # garbles Hindi/Hinglish audio into nonsense English-phonetic text (e.g.
    # "mujhe lagta hai koi meri baat samajhta hi nahi" transcribed as
    # "Mufalad de Haykoy Meribat Samashtra High" — confirmed live). Compared
    # "en" (default), "multi" (nova-3 code-switching), and "hi" directly
    # against the same synthesized Hindi audio: "hi" was dramatically more
    # accurate ("mujhe meri baat samajh..." correctly in Devanagari) vs
    # "multi"'s still-garbled romanized guess and "en"'s complete gibberish.
    # Only reached when Sarvam isn't configured — this is the
    # degraded-but-still-functional fallback path, not the primary Hindi path.
    stt_language = "hi" if session_language in ("hi", "hinglish") else "en"
    return (
        DeepgramSTTStream(sample_rate=16000, language=stt_language),
        DeepgramTTSStream(sample_rate=24000),
    )


class VoiceSession:
    """Coordinates STT, TTS, and barge-in state for one voice call."""

    def __init__(
        self,
        websocket: WebSocket,
        db: AsyncSession,
        user_id: str,
        session_id: str,
        session_language: str = "en",
    ):
        self.ws = websocket
        self.db = db
        self.user_id = user_id
        self.session_id = session_id
        self.session_language = session_language
        self.stt, self.tts = _select_voice_provider(session_language)
        self.is_ai_speaking = False
        self._turn_task: Optional[asyncio.Task] = None
        self._ducked = False
        self._unduck_task: Optional[asyncio.Task] = None
        self._audio_buffer = bytearray()
        # StrategyNode move history — tracked per-call (one VoiceSession per
        # WS connection). Doesn't persist across separate calls to the same
        # session_id, unlike text-chat's DB-backed history; acceptable since
        # a voice call is typically one continuous conversation.
        self._last_three_moves: list[str] = []
        self._relevant_context: str = ""
        self._last_mood: str = "neutral"

    async def start(self) -> bool:
        stt_ok = await self.stt.start()
        tts_ok = await self.tts.start()
        return stt_ok and tts_ok

    def append_audio(self, pcm16_bytes: bytes) -> None:
        self._audio_buffer.extend(pcm16_bytes)
        if len(self._audio_buffer) > _AUDIO_BUFFER_MAX_BYTES:
            del self._audio_buffer[: len(self._audio_buffer) - _AUDIO_BUFFER_MAX_BYTES]

    async def send_json(self, payload: dict) -> None:
        await self.ws.send_text(json.dumps(payload))

    async def _cancel_turn_task(self) -> None:
        """Cancel and fully await the in-flight turn before starting another —
        the two must never touch self.db concurrently (AsyncSession isn't
        safe for concurrent use from multiple coroutines)."""
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass
            # A turn cancelled mid-query (barge-in, or a new turn starting)
            # leaves the AsyncSession with a broken transaction. Every later
            # turn then fails at its FIRST query with PendingRollbackError
            # ("Can't reconnect until invalid transaction is rolled back") —
            # observed live 2026-08-08: turns 1-2 fine, then every remaining
            # turn of the call returned "Processing failed". Rolling back
            # here, outside the cancellation context, restores the session so
            # an interruption costs one turn instead of the whole call.
            await self._safe_rollback()

    async def _safe_rollback(self) -> None:
        """Best-effort transaction reset. Never raises: it runs on paths that
        are already handling a failure, and a rollback that itself throws must
        not mask the original problem or kill the call."""
        try:
            await self.db.rollback()
        except Exception as exc:
            logger.warning(
                "DB rollback failed | session=%s | %s", self.session_id, exc
            )

    def _turn_active(self) -> bool:
        """True from the moment a turn is triggered until it fully finishes —
        a strictly wider window than is_ai_speaking, which only flips True
        once sentiment+strategy+RAG have resolved and TTS actually starts.
        Needed so barge-in also covers the pre-speech processing window (see
        handle_barge_in)."""
        return self._turn_task is not None and not self._turn_task.done()

    async def duck_for_possible_speech(self) -> None:
        """Stage 1 of barge-in, driven by raw mic energy (SpeechStarted).

        Energy alone is unreliable — background noise, or our own TTS coming
        back through the user's speakers, both trip it. But waiting for
        transcribed words before reacting means talking over the user for
        ~1-2s, which feels awful. So we split it: energy instantly PAUSES
        playback (cheap and reversible), and only confirmed words tear the
        turn down. If no words materialise we silently resume.
        """
        if not self.is_ai_speaking or self._ducked:
            return
        self._ducked = True
        await self.send_json({"type": "duck"})
        if self._unduck_task and not self._unduck_task.done():
            self._unduck_task.cancel()
        self._unduck_task = asyncio.create_task(self._unduck_after_grace())

    async def _unduck_after_grace(self) -> None:
        try:
            await asyncio.sleep(BARGE_IN_CONFIRM_S)
        except asyncio.CancelledError:
            return
        if self._ducked and self.is_ai_speaking:
            self._ducked = False
            logger.info(
                "Ducked on mic energy but no speech transcribed — resuming | session=%s",
                self.session_id,
            )
            await self.send_json({"type": "unduck"})

    async def handle_barge_in(self) -> None:
        """Stage 2: real words confirmed — commit to the interruption.

        Gated on _turn_active(), not just is_ai_speaking. is_ai_speaking only
        flips True once sentiment+strategy+RAG resolve and TTS actually
        starts — there's a real multi-second window (worst case, LLM
        sentiment timeout) where a turn is in-flight but not yet speaking.
        Previously, real words transcribed during that window skipped
        barge-in entirely (gated on is_ai_speaking alone), silently piled
        into pending_transcript, and ~TURN_STALL_TIMEOUT_S later the stall
        watchdog fired a second, spurious start_turn that cancelled the
        still-in-flight first turn out from under it — observed live as two
        overlapping "Turn triggered" log lines (via=smart_turn then, ~10s
        later, via=stall_watchdog) for what should have been one turn.
        Widening the gate here means new words during that window are
        handled immediately, as a normal barge-in, instead of resurfacing
        later as a confusing phantom restart.
        """
        if self._unduck_task and not self._unduck_task.done():
            self._unduck_task.cancel()
        self._ducked = False
        was_speaking = self.is_ai_speaking
        if not was_speaking and not self._turn_active():
            return
        logger.info(
            "Barge-in confirmed | session=%s | was_speaking=%s",
            self.session_id, was_speaking,
        )
        self.is_ai_speaking = False
        await self._cancel_turn_task()
        if was_speaking:
            await self.tts.cancel()
        await self.send_json({"type": "interrupted"})

    async def pump_stt_events(self) -> None:
        """Consumes STT events; triggers barge-in and starts turns via a
        hybrid acoustic+semantic turn-detector.

        Deepgram's UtteranceEnd (~1000ms+ of silence, elapsed-audio-time
        based) is the ever-present fallback — it always eventually fires and
        always starts a turn if there's pending text. But Deepgram also
        reports speech_final (~300ms, endpointing-based) on individual final
        segments well before UtteranceEnd — a much faster *acoustic* signal
        that's prone to false positives on ordinary mid-thought pauses. To
        act on it safely, each speech_final is paired with a fast *semantic*
        completeness check (is_utterance_complete) run on the accumulated
        text so far; only if both agree does a turn start early.

        A third signal, Smart Turn v3 (turn_detector.py — a standalone,
        audio-based turn-completion model, not text-based like the semantic
        check), runs in parallel on the same speech_final trigger: whichever
        of it or the semantic check confirms completeness first wins. It's
        faster (~12-65ms local CPU vs ~150-450ms for a Groq round-trip) but
        empirically noisier on some utterances (verified against our own
        fixtures — see turn_detector.py docstring), so it's used as an
        additional fast path, not a replacement.

        This can only make turn-taking faster, never worse: if either check
        is wrong and the user keeps talking, that's indistinguishable from
        an interruption and is caught by the barge-in path the moment their
        next words are transcribed.

        A stall watchdog backs the whole thing up, so a failure of ALL
        triggers degrades to a late reply rather than no reply at all.
        """
        pending_transcript = ""
        semantic_check_epoch = 0
        semantic_check_in_flight = False
        smart_turn_check_in_flight = False
        stall_task: Optional[asyncio.Task] = None

        def cancel_stall() -> None:
            nonlocal stall_task
            task, stall_task = stall_task, None
            if task and not task.done():
                task.cancel()

        async def start_turn(text: str, trigger: str) -> None:
            nonlocal pending_transcript
            text = text.strip()
            if not text:
                return
            logger.info(
                "Turn triggered | session=%s | via=%s | text=%r",
                self.session_id, trigger, text,
            )
            pending_transcript = ""
            cancel_stall()
            await self._cancel_turn_task()
            self._turn_task = asyncio.create_task(self._run_turn(text))

        async def stall_watchdog() -> None:
            """Last-resort guarantee that we always eventually respond.

            Both normal triggers can fail to arrive: the semantic check may
            say "incomplete", and UtteranceEnd depends on Deepgram observing
            real silence — which upstream audio problems (an over-aggressive
            mic gate, continuous background noise) can prevent indefinitely.
            Without this, those cases present as the agent simply never
            replying, which is the worst possible failure for a support tool.
            """
            nonlocal stall_task
            try:
                await asyncio.sleep(TURN_STALL_TIMEOUT_S)
            except asyncio.CancelledError:
                return
            stall_task = None
            text = pending_transcript.strip()
            if text:
                logger.warning(
                    "Turn-detection stalled %.1fs with pending speech — firing "
                    "fallback turn | session=%s",
                    TURN_STALL_TIMEOUT_S, self.session_id,
                )
                await start_turn(text, "stall_watchdog")

        def arm_stall() -> None:
            nonlocal stall_task
            cancel_stall()
            stall_task = asyncio.create_task(stall_watchdog())

        async def check_semantic_completion(text: str, epoch: int) -> None:
            nonlocal semantic_check_in_flight
            try:
                complete = await sentiment_service.is_utterance_complete(text)
            finally:
                semantic_check_in_flight = False
            # Stale if a newer segment or UtteranceEnd has since taken over —
            # act only if nothing has changed since this check was fired off.
            if complete and epoch == semantic_check_epoch and text == pending_transcript.strip():
                await start_turn(text, "semantic")

        async def check_smart_turn_completion(audio_snapshot: bytes, text: str, epoch: int) -> None:
            nonlocal smart_turn_check_in_flight
            try:
                complete = await smart_turn_detector.is_utterance_complete(audio_snapshot)
            finally:
                smart_turn_check_in_flight = False
            # complete is True / False / None ("no opinion" — model unavailable
            # or inference failed). Only act on a confident True; None or
            # False both just defer to the other signals.
            if complete and epoch == semantic_check_epoch and text == pending_transcript.strip():
                await start_turn(text, "smart_turn")

        try:
            async for event_type, payload in self.stt.events():
                if event_type == STT_TRANSCRIPT and payload.transcript:
                    # Barge-in on actual transcribed WORDS, not raw VAD energy.
                    # SpeechStarted is energy-based, so background noise — or
                    # our own TTS bleeding from the user's speakers back into
                    # the mic — can trip it and kill a reply mid-sentence.
                    # Requiring words costs little (interim results arrive
                    # within ~100-300ms) and removes that whole class of
                    # spurious interruption.
                    await self.handle_barge_in()

                    if payload.is_final:
                        pending_transcript = (pending_transcript + " " + payload.transcript).strip()
                        arm_stall()
                    await self.send_json({
                        "type": "final_transcript" if payload.is_final else "partial_transcript",
                        "text": payload.transcript,
                    })
                    # Skip firing a new check of either kind while one of that
                    # kind is already in flight — a burst of close-together
                    # speech_final segments (common with natural mid-thought
                    # pauses) would otherwise fire one call per segment. The
                    # next speech_final or the UtteranceEnd fallback will
                    # still catch it either way.
                    if payload.is_final and payload.speech_final and pending_transcript:
                        semantic_check_epoch += 1
                        epoch = semantic_check_epoch
                        if not semantic_check_in_flight and hasattr(sentiment_service, "is_utterance_complete"):
                            semantic_check_in_flight = True
                            asyncio.create_task(
                                check_semantic_completion(pending_transcript, epoch)
                            )
                        if not smart_turn_check_in_flight:
                            smart_turn_check_in_flight = True
                            asyncio.create_task(
                                check_smart_turn_completion(
                                    bytes(self._audio_buffer), pending_transcript, epoch
                                )
                            )

                elif event_type == STT_SPEECH_STARTED:
                    # Stage 1 only: pause output now, decide later.
                    await self.duck_for_possible_speech()

                elif event_type == STT_UTTERANCE_END:
                    semantic_check_epoch += 1  # invalidate any in-flight check
                    await start_turn(pending_transcript, "utterance_end")

                elif event_type == STT_ERROR:
                    # Recoverable: the STT stream marks itself dead on error
                    # and reconnects on the next audio packet (~20ms), so a
                    # dropped socket costs a moment of hearing, not the call.
                    # Surfacing a hard "Speech recognition error" here made a
                    # self-healing blip look terminal to the user, who had no
                    # way to know it had already recovered.
                    logger.warning(
                        "STT error — reconnecting | session=%s", self.session_id
                    )
                    await self.send_json({"type": "stt_reconnecting"})
        finally:
            cancel_stall()

    async def pump_tts_audio(self) -> None:
        """Forwards TTS audio bytes to the client as binary WS frames."""
        async for chunk in self.tts.audio_chunks():
            try:
                await self.ws.send_bytes(chunk)
            except Exception:
                return

    async def _run_turn(self, message: str) -> None:
        """One conversation turn: sentiment -> crisis-or-RAG+LLM -> TTS."""
        t_start = time.monotonic()
        try:
            _, session = await _get_or_create_user_session(self.db, self.user_id, self.session_id)
            self.db.add(ChatMessage(session_id=session.id, role="user", content=message))
            await self.db.commit()

            longitudinal_context = await _load_longitudinal_context(
                self.db, self.user_id, self.session_id
            )
            sentiment_state = {
                "messages": [{"role": "user", "content": message}],
                "user_id": self.user_id,
            }
            # Patient-memory recall runs INSIDE this gather, not after it, so
            # it costs no extra wall-clock — it overlaps the sentiment LLM
            # round-trip, which dominates this stage anyway. Sequencing it
            # would have added an embedding + vector search to every turn's
            # critical path, which is exactly the mistake that made
            # retrieve_style_exemplars so expensive before it was cached.
            # retrieve_patient_memory hard-filters on user_id and returns []
            # when the collection does not exist, so this degrades to a no-op
            # rather than an error where the store was never built.
            sentiment_result, history, past_memories = await asyncio.wait_for(
                asyncio.gather(
                    _sentiment_node(sentiment_state),
                    _load_history(self.db, self.session_id),
                    rag_service.retrieve_patient_memory(
                        user_id=self.user_id, query=message, k=3
                    ),
                ),
                timeout=LLM_SENTIMENT_TIMEOUT_S,
            )
            t_sentiment_done = time.monotonic()
            mood = sentiment_result["current_mood"]
            self._last_mood = mood
            risk_score = sentiment_result["risk_score"]
            language = sentiment_result["detected_language"]
            cognitive_distortion = sentiment_result["cognitive_distortion_detected"]
            is_crisis = sentiment_result["is_crisis"]
            level = _risk_level(risk_score)

            # Episodic layer: one row per turn, not per session. MemoryService
            # .record_turn existed but was never called from any route, so
            # mood_trajectory sat empty (0 rows in prod) and every
            # trajectory-based feature — slope detection, dependency
            # monitoring, recurrence recall — had nothing to read. Recorded
            # here, right after sentiment resolves, because this is the only
            # point where mood and risk are both known for THIS turn.
            # Deliberately not fatal: a trajectory write must never cost the
            # user their reply.
            try:
                await MemoryService(self.db).record_turn(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    turn_index=len(history) + 1,
                    mood=mood,
                    risk_score=risk_score,
                )
            except Exception as exc:
                logger.warning(
                    "Trajectory record failed | session=%s | %s", self.session_id, exc
                )

            await self.send_json({"type": "meta", "mood": mood, "risk_score": risk_score})

            full_response = ""
            self.is_ai_speaking = True
            await self.send_json({"type": "speaking_started"})

            if is_crisis:
                full_response = crisis_message_for(language)
                await self.tts.speak(full_response)
            else:
                # Same move selection text-chat's StrategyNode uses. `history`
                # (loaded above from the DB, same as text-chat) already
                # includes the current turn's user message — reused here so
                # turn_index (StrategyNode counts user messages in `messages`)
                # actually grows turn over turn. Previously this passed a
                # fresh single-item list every turn, pinning turn_index at 1
                # for the whole call — which permanently suppresses
                # psychoeducation AND open_question (the turn_index<=2 rule),
                # not just on the opening turn. Confirmed live: 7 consecutive
                # turns in one call, zero open_question/psychoeducation
                # selections, exactly matching the bug.
                strategy_state = {
                    "risk_score": risk_score,
                    "current_mood": mood,
                    "cognitive_distortion_detected": cognitive_distortion,
                    "last_three_moves": self._last_three_moves,
                    "messages": history,
                    "relevant_context": self._relevant_context,
                    "user_id": self.user_id,
                }
                strategy_result = await _strategy_node(strategy_state)
                move = strategy_result["selected_move"]
                self._last_three_moves = strategy_result["last_three_moves"]

                needs_rag = _needs_rag(message, mood, move)
                if needs_rag:
                    # Language-aware backchannel ack
                    acks = _BACKCHANNELS_HI if self.session_language in ("hi", "hinglish") else _BACKCHANNELS
                    await self.tts.speak(random.choice(acks))
                    context = await rag_service.retrieve_clinical_context(message, mood)
                else:
                    context = ""
                self._relevant_context = context
                # Three distinct memory sources, kept under separate headings
                # rather than concatenated: they carry different epistemic
                # weight and the model must not treat a recalled fragment of
                # the user's own past speech as clinical evidence.
                #   [Previous sessions]  — summaries of whole past sessions
                #   [You may recall]     — semantic recall of this user's own
                #                          past dialogue (patient_memory)
                #   [Clinical evidence]  — retrieved clinical KB passages
                blocks = []
                if longitudinal_context:
                    blocks.append(f"[Previous sessions]\n{longitudinal_context}")
                if past_memories:
                    recalled = "\n".join(f"- {m}" for m in past_memories)
                    blocks.append(
                        "[You may recall from earlier conversations with this "
                        f"person]\n{recalled}"
                    )
                if context:
                    blocks.append(f"[Clinical evidence]\n{context}")
                merged_context = "\n\n".join(blocks)

                style_exemplars = await rag_service.retrieve_style_exemplars(
                    move=move,
                    affect_valence=_affect_valence_for_mood(mood),
                    register=_register_for_language(language),
                    k=2,
                )
                system_prompt = build_therapeutic_system_prompt(
                    context=merged_context,
                    mood=mood,
                    move=move,
                    language=language,
                    style_exemplars=style_exemplars,
                )

                history.append({"role": "user", "content": message})

                buf = ""
                flushed_first_sentence = False
                first_token_received = False
                # Manual iteration (not `async for`) so each step can carry
                # its own timeout — a tight one until the very first token
                # arrives (this is exactly where a rate-limited provider hangs
                # silently, see LLM_FIRST_TOKEN_TIMEOUT_S's docstring above),
                # a shared overall budget after that so a normal multi-second
                # stream isn't mistaken for a hang.
                stream_start = time.monotonic()
                # therapy_llm_service, NOT sentiment_service — generation
                # keeps the quality-first (OpenRouter->Gemini->Groq) provider
                # order; sentiment_service is now Groq-first for latency
                # (see workflow.py) and using it here would apply that same
                # fast-but-lower-quality ordering to response text too.
                token_iter = therapy_llm_service.stream_response_for_move(
                    history=history, system_prompt=system_prompt
                ).__aiter__()
                while True:
                    elapsed = time.monotonic() - stream_start
                    remaining = LLM_STREAM_TOTAL_TIMEOUT_S - elapsed
                    if remaining <= 0:
                        raise TimeoutError("Voice generation stream exceeded total time budget")
                    step_timeout = LLM_FIRST_TOKEN_TIMEOUT_S if not first_token_received else remaining
                    try:
                        token = await asyncio.wait_for(token_iter.__anext__(), timeout=step_timeout)
                    except StopAsyncIteration:
                        break
                    first_token_received = True
                    full_response += token
                    buf += token
                    sentences, buf = _flush_sentences(buf)
                    for sentence in sentences:
                        await self.tts.send_text(sentence)
                        if not flushed_first_sentence:
                            # Flush once, early, so audio starts promptly —
                            # Deepgram won't start synthesizing a single short
                            # sentence on its own (confirmed empirically).
                            # Sentences after this one are NOT flushed
                            # individually, so Deepgram keeps generating
                            # continuously instead of gapping between them.
                            await self.tts.flush()
                            flushed_first_sentence = True
                if buf.strip():
                    await self.tts.send_text(buf.strip())
                await self.tts.flush()

            # speak() only queues text with Deepgram and returns in ~ms — it
            # does not wait for audio to actually finish generating/streaming.
            # Without this wait, is_ai_speaking flips back to False almost
            # immediately, making barge-in a no-op for virtually the entire
            # time audio is actually playing.
            await self.tts.wait_until_flushed()
            self.is_ai_speaking = False
            await self.send_json({"type": "speaking_ended"})

            t_done = time.monotonic()
            logger.info(
                "Turn timing | session=%s | crisis=%s | pre_sentiment=%.0fms | "
                "response_and_tts=%.0fms | total=%.0fms",
                self.session_id, is_crisis,
                (t_sentiment_done - t_start) * 1000,
                (t_done - t_sentiment_done) * 1000,
                (t_done - t_start) * 1000,
            )

            # selected_move persisted so response-quality problems are
            # diagnosable straight from the DB. It was always NULL for voice,
            # which meant tracing "why did it stop asking questions?" required
            # cross-referencing server logs by timestamp.
            self.db.add(ChatMessage(
                session_id=session.id, role="assistant", content=full_response,
                detected_mood=mood, selected_move=(None if is_crisis else move),
            ))
            if level != "LOW":
                session.risk_level = level
                self.db.add(session)
            await self.db.commit()

            await self.send_json({"type": "done", "risk_level": level})

        except asyncio.CancelledError:
            self.is_ai_speaking = False
            raise
        except Exception as e:
            logger.error("Voice turn error | session=%s | %s", self.session_id, str(e), exc_info=True)
            # Clear any half-open transaction so this failure costs one turn
            # rather than permanently breaking the session for the rest of
            # the call (see _cancel_turn_task).
            await self._safe_rollback()
            # Previously this only sent a JSON "error" event — no audio at
            # all, indistinguishable from the call just going dead to anyone
            # not inspecting WS control frames. On a voice call the failure
            # itself needs to be spoken, not just logged.
            language = _detect_language(message)
            fallback = (
                "Maaf karna, mujhe thoda sa dikkat aa rahi hai abhi. Thoda ruk kar phir se try karo."
                if language in ("hi", "hinglish")
                else "I'm having a little trouble right now — give me just a moment and try again."
            )
            try:
                self.is_ai_speaking = True
                await self.send_json({"type": "speaking_started"})
                await self.tts.speak(fallback)
                await self.tts.wait_until_flushed()
            except Exception:
                pass  # best-effort — still send error/done below either way
            finally:
                self.is_ai_speaking = False
                await self.send_json({"type": "speaking_ended"})
            await self.send_json({"type": "error", "message": "Processing failed"})
            await self.send_json({"type": "done", "risk_level": "LOW"})

    async def close(self) -> None:
        if self._unduck_task and not self._unduck_task.done():
            self._unduck_task.cancel()
        await self._cancel_turn_task()
        await self.stt.close()
        await self.tts.close()

    async def consolidate_session(self) -> None:
        """Session-end structured-memory consolidation — mirrors text-chat's
        periodic SummaryNode trigger (every SUMMARY_INTERVAL messages), but a
        voice call has no such natural mid-call checkpoint (it's typically
        one continuous conversation), so WS disconnect is the only sensible
        trigger point here. Best-effort: extract_session_insights/
        apply_consolidation already swallow their own errors and degrade to
        "nothing to consolidate" rather than raising, but wrapped anyway
        since this runs during teardown and must never block the WS from
        closing cleanly."""
        try:
            history = await _load_history(self.db, self.session_id, limit=40)
            insights = await extract_session_insights(
                therapy_llm_service, history, self._last_mood
            )
            if insights:
                await MemoryService(self.db).apply_consolidation(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    insights=insights,
                    rag_service=rag_service,
                )
        except Exception as e:
            logger.error(
                "Voice session consolidation failed | session=%s | %s",
                self.session_id, e,
            )


@router.websocket("/ws/voice/{session_id}")
async def websocket_voice(
    websocket: WebSocket,
    session_id: str,
    token: str = "",
    lang: str = "en",   # client hints detected language: en | hi | hinglish
    db: AsyncSession = Depends(get_db),
):
    """
    Auth mirrors /ws/chat: browsers can't set an Authorization header on the
    WS handshake, so the JWT is passed as a query param.
    """
    from jose import JWTError
    from app.core.security import decode_token

    if not voice_enabled():
        await websocket.accept()
        await websocket.close(code=4503, reason="Voice is not configured on this server")
        return

    try:
        authenticated_user_id = decode_token(token)
    except JWTError:
        await websocket.accept()
        await websocket.close(code=4401, reason="Invalid or missing token")
        return

    await websocket.accept()
    logger.info("Voice WebSocket connected | session=%s | user=%s | lang=%s", session_id, authenticated_user_id, lang)

    voice_session = VoiceSession(websocket, db, authenticated_user_id, session_id, session_language=lang)
    if not await voice_session.start():
        await websocket.send_text(json.dumps({"type": "error", "message": "Voice provider unavailable"}))
        await websocket.close(code=4502, reason="Could not start voice provider")
        return

    stt_task = asyncio.create_task(voice_session.pump_stt_events())
    tts_task = asyncio.create_task(voice_session.pump_tts_audio())
    await voice_session.send_json({"type": "ready"})

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await voice_session.stt.send_audio(message["bytes"])
                voice_session.append_audio(message["bytes"])
            # Text frames reserved for future control messages (e.g. mute) — ignored in V1.
    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected | session=%s", session_id)
    finally:
        stt_task.cancel()
        tts_task.cancel()
        await voice_session.close()
        await voice_session.consolidate_session()
