import json
import asyncio
import re
import time
from fastapi import APIRouter, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.database import get_db
from app.infrastructure.models import User, ChatSession, ChatMessage
from app.domain.state import ChatInput, ChatOutput
import app.graph.workflow as _workflow
from app.graph.workflow import sentiment_service, gemini_service, rag_service, therapy_llm_service
from app.services.memory_service import extract_session_insights, MemoryService
from app.api.dependencies import get_current_user
from app.api.limiter import limiter
from app.core.logging import logging
from app.graph.nodes.sentiment import SentimentNode, _build_multimodal_hint, _detect_language
from app.graph.nodes.strategy import StrategyNode
from app.graph.nodes.therapy import _needs_rag, _affect_valence_for_mood, _register_for_language
from app.graph.nodes.crisis import crisis_message_for
from app.services.crisis_rules import apply_crisis_floor
from app.services.therapeutic_prompt import build_therapeutic_system_prompt

# Web-chat WebSocket pipeline nodes — same construction voice.py uses, so the
# text and voice experiences select moves and score risk identically.
_ws_sentiment_node = SentimentNode(sentiment_service, fallback_llm_service=gemini_service)
_ws_strategy_node = StrategyNode()

# Upper bound on how long a turn waits for clinical retrieval before replying
# without grounding (typical 1.4s; one 15s outlier measured).
WS_RAG_TIMEOUT_S = 4.0

router = APIRouter()
logger = logging.getLogger(__name__)

CRISIS_MESSAGE = (
    "I'm concerned about what you're going through. I am an AI, not a human, "
    "and I want to make sure you're safe. Please contact a local emergency service "
    "or a suicide prevention hotline immediately."
)


# ---------------------------------------------------------------------------
# Shared DB helper
# ---------------------------------------------------------------------------

async def _get_or_create_user_session(
    db: AsyncSession, user_id: str, session_id: str
) -> tuple:
    """Return (user, session), creating them if they don't exist yet."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    if not user:
        user = User(id=user_id, email=f"{user_id}@example.com")
        db.add(user)
        await db.commit()

    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalars().first()
    if not session:
        session = ChatSession(id=session_id, user_id=user.id)
        db.add(session)
        await db.commit()

    return user, session


async def _load_history(db: AsyncSession, session_id: str, limit: int = 10) -> list:
    """Load the last `limit` messages for a session as role/content dicts."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(limit)
    )
    msgs = list(reversed(result.scalars().all()))
    return [{"role": m.role, "content": m.content} for m in msgs]


async def _load_longitudinal_context(
    db: AsyncSession, user_id: str, current_session_id: str, limit: int = 3
) -> str:
    """
    Return a condensed view of the user's recent past sessions.
    Provides the LLM with mood trajectory and recurring themes across visits.
    Only sessions that have a non-empty summary are included.
    """
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .where(ChatSession.id != current_session_id)
        .where(ChatSession.summary.isnot(None))
        .order_by(ChatSession.created_at.desc())
        .limit(limit)
    )
    sessions = list(reversed(result.scalars().all()))  # oldest first
    if not sessions:
        return ""
    parts = [f"[{s.risk_level}] {s.summary}" for s in sessions]
    return " → ".join(parts)


async def _record_trajectory(
    db, user_id: str, session_id: str, mood: str, risk_score: int
) -> None:
    """Append this turn to the episodic mood time-series.

    MemoryService.record_turn existed but had no caller anywhere in the app,
    so mood_trajectory stayed empty (0 rows in production) and everything
    built on it — slope detection, dependency monitoring, recurrence recall —
    silently had nothing to read. Called from every path that resolves a
    mood/risk pair so the series does not develop holes depending on which
    endpoint a client happens to use.

    turn_index is derived from the row count rather than passed in: the
    endpoints here compute message counts differently (some include the
    assistant turn, some do not), and a consistent monotonic index matters
    more for slope detection than matching any one endpoint's notion of a
    turn. Never fatal — trajectory is analytics, not the user's reply.
    """
    try:
        from sqlalchemy import func as _func
        from app.infrastructure.models import MoodTrajectory
        result = await db.execute(
            select(_func.count(MoodTrajectory.id)).where(
                MoodTrajectory.session_id == session_id
            )
        )
        turn_index = (result.scalar() or 0) + 1
        await MemoryService(db).record_turn(
            user_id=user_id,
            session_id=session_id,
            turn_index=turn_index,
            mood=mood,
            risk_score=risk_score,
        )
    except Exception as e:
        logger.warning("Trajectory record failed | session=%s | %s", session_id, e)


def _risk_level(score: int) -> str:
    if score >= 8:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# POST /chat  — standard blocking endpoint (existing behaviour)
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatOutput)
@limiter.limit("20/minute")
async def chat_endpoint(
    request: Request,
    payload: ChatInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info("Chat Request: %s | Session: %s", payload.user_id, payload.session_id)

    try:
        _, session = await _get_or_create_user_session(db, payload.user_id, payload.session_id)

        user_msg = ChatMessage(
            session_id=session.id, role="user", content=payload.message
        )
        db.add(user_msg)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error("Database Error (Pre-Chat): %s", e)
        raise HTTPException(status_code=500, detail="Database write failed")

    # Load past-session summaries for longitudinal mood awareness.
    longitudinal_context = await _load_longitudinal_context(
        db, payload.user_id, payload.session_id
    )

    config = {"configurable": {"thread_id": payload.session_id}}
    initial_state = {
        "messages": [{"role": "user", "content": payload.message}],
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "longitudinal_context": longitudinal_context or None,
        "audio_features": payload.audio_features.model_dump() if payload.audio_features else None,
        "video_features": payload.video_features.model_dump() if payload.video_features else None,
    }

    try:
        final_state = await _workflow.psych_graph.ainvoke(initial_state, config=config)

        last_message = final_state["messages"][-1]
        ai_response_content = (
            last_message["content"]
            if isinstance(last_message, dict)
            else last_message.content
        )

        detected_mood = final_state.get("current_mood", "neutral")
        risk_score = final_state.get("risk_score", 0)
        level = _risk_level(risk_score)
        await _record_trajectory(db, payload.user_id, payload.session_id, detected_mood, risk_score)

        ai_msg = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=ai_response_content,
            detected_mood=detected_mood,
        )
        db.add(ai_msg)

        session_dirty = False
        if level != "LOW":
            session.risk_level = level
            session_dirty = True
        # Persist summary generated by SummaryNode (every 10 messages).
        new_summary = final_state.get("session_summary")
        if new_summary:
            session.summary = new_summary
            session_dirty = True
            logger.info("Session summary persisted | session=%s", session.id)
            # Same checkpoint (every SUMMARY_INTERVAL messages) also runs the
            # deeper structured-entity consolidation pass — piggybacking on
            # SummaryNode's existing trigger condition rather than adding a
            # second, separate message-count check.
            try:
                insights = await extract_session_insights(
                    therapy_llm_service, final_state["messages"], detected_mood
                )
                if insights:
                    await MemoryService(db).apply_consolidation(
                        user_id=payload.user_id,
                        session_id=session.id,
                        insights=insights,
                        rag_service=rag_service,
                    )
            except Exception as e:
                logger.error("Session consolidation failed | session=%s | %s", session.id, e)
        if session_dirty:
            db.add(session)
        await db.commit()

        return ChatOutput(
            response=ai_response_content,
            detected_mood=detected_mood,
            risk_level=level,
        )

    except Exception as e:
        logger.error("AI/Graph Error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="AI processing failed")


# ---------------------------------------------------------------------------
# POST /chat/stream  — SSE streaming endpoint for audio / video agents
# ---------------------------------------------------------------------------
#
# Event format (newline-delimited SSE):
#   data: {"type": "meta",  "mood": "anxious", "risk_score": 4}
#   data: {"type": "token", "content": "I hear you..."}
#   data: {"type": "token", "content": " that sounds difficult."}
#   data: {"type": "done",  "risk_level": "MEDIUM"}
#
# Audio/video clients: buffer tokens until a sentence boundary
# (`. `, `! `, `? `) then pass the sentence to TTS / avatar.
# ---------------------------------------------------------------------------

@router.post("/chat/stream")
@limiter.limit("10/minute")
async def stream_chat_endpoint(
    request: Request,
    payload: ChatInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Optimised streaming path:
      1. Sentiment via Groq      (~150 ms, fast JSON)
      2. RAG retrieval           (parallel MMR + reranking)
      3. Gemini streams tokens   (client receives first token in ~1 s)

    Parallel execution: steps 1 and 2 run concurrently — sentiment and
    raw-query RAG search overlap, so the mood-expanded reranking is the
    only sequential dependency.
    """
    logger.info("Stream Request: %s | Session: %s", payload.user_id, payload.session_id)

    async def event_stream():
        try:
            # -- DB setup --------------------------------------------------
            _, session = await _get_or_create_user_session(
                db, payload.user_id, payload.session_id
            )
            user_msg = ChatMessage(
                session_id=session.id, role="user", content=payload.message
            )
            db.add(user_msg)
            await db.commit()

            # -- Step 1: Sentiment + history load (parallel) ---------------
            analysis, history = await asyncio.gather(
                sentiment_service.analyze_sentiment(payload.message),
                _load_history(db, payload.session_id),
            )
            mood = analysis.get("mood", "neutral")
            risk_score = int(analysis.get("risk_score", 0))
            # Deterministic floor: the model may miss indirect cues (see crisis_rules.py).
            risk_score = apply_crisis_floor(payload.message, risk_score, source="chat_stream")
            language = _detect_language(payload.message)
            is_crisis = risk_score >= 8
            level = _risk_level(risk_score)
            await _record_trajectory(db, payload.user_id, payload.session_id, mood, risk_score)

            # Send mood/risk immediately so clients can react early
            yield f"data: {json.dumps({'type': 'meta', 'mood': mood, 'risk_score': risk_score})}\n\n"

            # -- Step 2: Route & respond -----------------------------------
            full_response = ""

            if is_crisis:
                full_response = crisis_message_for(language)
                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
            else:
                # Skip RAG for short/trivial messages (< 4 words, non-clinical mood)
                # — saves 1.5-4s of cross-encoder CPU time before first token.
                _clinical = {"anxious", "depressed", "lonely", "angry", "stressed",
                             "fearful", "hopeless", "guilty", "confused"}
                if len(payload.message.split()) >= 4 or mood in _clinical:
                    context = await rag_service.retrieve_clinical_context(
                        payload.message, mood
                    )
                else:
                    context = ""

                # Add current message to history for generation context
                history.append({"role": "user", "content": payload.message})

                # Stream tokens — Groq when GROQ_API_KEY is set (~100ms to first
                # token); Gemini fallback otherwise (sentiment_service = gemini_service).
                async for token in sentiment_service.stream_therapeutic_response(
                    history=history, context=context, mood=mood, language=language
                ):
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # -- Step 3: Done event + DB persist ---------------------------
            yield f"data: {json.dumps({'type': 'done', 'risk_level': level})}\n\n"

            ai_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=full_response,
                detected_mood=mood,
            )
            db.add(ai_msg)
            if level != "LOW":
                session.risk_level = level
                db.add(session)
            await db.commit()

        except Exception as e:
            logger.error("Stream Error: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx proxy buffering
        },
    )


# ---------------------------------------------------------------------------
# Sentence-boundary helper
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"([.!?])\s")


def _flush_sentences(buf: str) -> tuple[list[str], str]:
    """
    Split *buf* on sentence boundaries (`. `, `! `, `? `).
    Returns (list_of_complete_sentences, leftover_fragment).
    """
    sentences: list[str] = []
    pos = 0
    for m in _SENTENCE_END.finditer(buf):
        end = m.end()
        sentence = buf[pos:end].strip()
        if sentence:
            sentences.append(sentence)
        pos = end
    return sentences, buf[pos:]


# ---------------------------------------------------------------------------
# POST /chat/stream/sentences  — sentence-level SSE for TTS / avatar clients
# ---------------------------------------------------------------------------
#
# Event format:
#   data: {"type": "meta",     "mood": "anxious", "risk_score": 4}
#   data: {"type": "sentence", "content": "I hear you, that sounds difficult."}
#   data: {"type": "sentence", "content": "Let's try a breathing exercise."}
#   data: {"type": "done",     "risk_level": "MEDIUM"}
#
# TTS clients can pipe each "sentence" event directly to speech synthesis
# without any client-side buffering logic.
# ---------------------------------------------------------------------------

@router.post("/chat/stream/sentences")
@limiter.limit("10/minute")
async def stream_sentences_endpoint(
    request: Request,
    payload: ChatInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Same pipeline as /chat/stream but buffers tokens server-side and emits
    one SSE event per complete sentence.  Ideal for TTS and avatar pipelines
    that need sentence-granularity without client-side buffering logic.
    """
    logger.info("Stream/Sentences Request: %s | Session: %s", payload.user_id, payload.session_id)

    async def sentence_stream():
        try:
            _, session = await _get_or_create_user_session(
                db, payload.user_id, payload.session_id
            )
            user_msg = ChatMessage(
                session_id=session.id, role="user", content=payload.message
            )
            db.add(user_msg)
            await db.commit()

            analysis, history = await asyncio.gather(
                sentiment_service.analyze_sentiment(payload.message),
                _load_history(db, payload.session_id),
            )
            mood = analysis.get("mood", "neutral")
            risk_score = int(analysis.get("risk_score", 0))
            # Deterministic floor: the model may miss indirect cues (see crisis_rules.py).
            risk_score = apply_crisis_floor(payload.message, risk_score, source="chat_stream")
            language = _detect_language(payload.message)
            is_crisis = risk_score >= 8
            level = _risk_level(risk_score)
            await _record_trajectory(db, payload.user_id, payload.session_id, mood, risk_score)

            yield f"data: {json.dumps({'type': 'meta', 'mood': mood, 'risk_score': risk_score})}\n\n"

            full_response = ""

            if is_crisis:
                full_response = crisis_message_for(language)
                yield f"data: {json.dumps({'type': 'sentence', 'content': full_response})}\n\n"
            else:
                _clinical = {"anxious", "depressed", "lonely", "angry", "stressed",
                             "fearful", "hopeless", "guilty", "confused"}
                if len(payload.message.split()) >= 4 or mood in _clinical:
                    context = await rag_service.retrieve_clinical_context(payload.message, mood)
                else:
                    context = ""

                history.append({"role": "user", "content": payload.message})

                buf = ""
                async for token in sentiment_service.stream_therapeutic_response(
                    history=history, context=context, mood=mood, language=language
                ):
                    full_response += token
                    buf += token
                    sentences, buf = _flush_sentences(buf)
                    for sentence in sentences:
                        yield f"data: {json.dumps({'type': 'sentence', 'content': sentence})}\n\n"

                # Flush any remaining fragment that never hit a sentence boundary
                if buf.strip():
                    yield f"data: {json.dumps({'type': 'sentence', 'content': buf.strip()})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'risk_level': level})}\n\n"

            ai_msg = ChatMessage(
                session_id=session.id, role="assistant",
                content=full_response, detected_mood=mood,
            )
            db.add(ai_msg)
            if level != "LOW":
                session.risk_level = level
                db.add(session)
            await db.commit()

        except Exception as e:
            logger.error("Stream/Sentences Error: %s", str(e), exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed'})}\n\n"

    return StreamingResponse(
        sentence_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# WS /ws/chat/{session_id}  — WebSocket for real-time voice agents
# ---------------------------------------------------------------------------
#
# Client sends JSON:
#   {"user_id": "abc", "message": "I feel anxious today",
#    "audio_features": {"tone": "trembling", "speech_rate": "fast", "energy": 0.8}}
#
# Server pushes JSON frames (same shape as SSE events):
#   {"type": "meta",     "mood": "anxious",   "risk_score": 5}
#   {"type": "sentence", "content": "I hear you..."}
#   {"type": "done",     "risk_level": "MEDIUM"}
#   {"type": "error",    "message": "..."}    (on failure)
#
# The connection stays open; clients send a new JSON payload for each turn.
# ---------------------------------------------------------------------------

@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    token: str = "",
    db: AsyncSession = Depends(get_db),
):
    """
    Persistent bidirectional WebSocket for real-time voice/avatar agents.
    Each text frame from the client is one conversation turn; the server
    streams sentence-level responses back over the same connection.

    Auth: browsers cannot set an Authorization header on the WS handshake,
    so the JWT is passed as a query param: /ws/chat/{session_id}?token=...
    """
    from jose import JWTError
    from app.core.security import decode_token

    try:
        authenticated_user_id = decode_token(token)
    except JWTError:
        # Accept first so the close code (4401) actually reaches the browser's
        # WebSocket.onclose handler — closing pre-accept only yields a generic
        # failed-handshake error with no code the client can act on.
        await websocket.accept()
        await websocket.close(code=4401, reason="Invalid or missing token")
        return

    await websocket.accept()
    logger.info("WebSocket connected | session=%s | user=%s", session_id, authenticated_user_id)

    # Per-connection conversational state for StrategyNode (voice keeps the
    # same two values on its session object).
    last_three_moves: list[str] = []
    relevant_context = ""

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            user_id = data.get("user_id", "")
            message = data.get("message", "").strip()
            audio_raw = data.get("audio_features")
            video_raw = data.get("video_features")

            if not user_id or not message:
                await websocket.send_text(json.dumps({"type": "error", "message": "user_id and message required"}))
                continue

            if user_id != authenticated_user_id:
                await websocket.send_text(json.dumps({"type": "error", "message": "user_id does not match authenticated token"}))
                continue

            try:
                t_turn = time.perf_counter()
                tm: dict[str, float] = {}          # per-stage ms, logged once per turn
                _, session = await _get_or_create_user_session(db, user_id, session_id)
                db.add(ChatMessage(session_id=session.id, role="user", content=message))
                await db.commit()
                tm["db_user_msg"] = (time.perf_counter() - t_turn) * 1000

                # Same calibrated sentiment path the graph and voice use
                # (passive-ideation scoring, language, cognitive distortion,
                # multimodal hint, and the deterministic crisis-rule net in
                # front of the model) instead of the bare analyze_sentiment().
                # Patient-memory recall joins the gather so it overlaps the
                # sentiment round-trip instead of adding to the critical path.
                # Only ONE query on the shared AsyncSession runs concurrently
                # here (_load_history) — a session is not safe for concurrent use.
                _t = time.perf_counter()
                sentiment_state = {
                    "messages": [{"role": "user", "content": message}],
                    "user_id": user_id,
                    "audio_features": audio_raw,
                    "video_features": video_raw,
                }
                sentiment_result, history, past_memories = await asyncio.gather(
                    _ws_sentiment_node(sentiment_state),
                    _load_history(db, session_id),
                    rag_service.retrieve_patient_memory(user_id=user_id, query=message, k=3),
                )
                tm["sentiment_history_memory"] = (time.perf_counter() - _t) * 1000
                mood = sentiment_result["current_mood"]
                risk_score = sentiment_result["risk_score"]
                language = sentiment_result["detected_language"]
                cognitive_distortion = sentiment_result["cognitive_distortion_detected"]
                is_crisis = sentiment_result["is_crisis"]
                level = _risk_level(risk_score)

                await websocket.send_text(
                    json.dumps({"type": "meta", "mood": mood, "risk_score": risk_score})
                )

                full_response = ""
                move: str | None = None

                if is_crisis:
                    full_response = crisis_message_for(language)
                    await websocket.send_text(
                        json.dumps({"type": "sentence", "content": full_response})
                    )
                else:
                    # `history` is loaded after this turn's user message was saved,
                    # so it already ends with it — appending it again sent the
                    # model the same message twice (and inflated StrategyNode's
                    # turn count by one).
                    if not (history and history[-1].get("role") == "user"
                            and history[-1].get("content") == message):
                        history.append({"role": "user", "content": message})

                    # Same move selection voice and the graph use. Previously the
                    # web chat was hard-wired to "simple_reflection" (via the
                    # legacy prompt), which never varies and never offers anything.
                    _t = time.perf_counter()
                    strategy_result = await _ws_strategy_node({
                        "risk_score": risk_score,
                        "current_mood": mood,
                        "cognitive_distortion_detected": cognitive_distortion,
                        "last_three_moves": last_three_moves,
                        "messages": history,
                        "relevant_context": relevant_context,
                        "user_id": user_id,
                    })
                    move = strategy_result["selected_move"]
                    last_three_moves = strategy_result["last_three_moves"]
                    tm["strategy"] = (time.perf_counter() - _t) * 1000

                    longitudinal_context = await _load_longitudinal_context(db, user_id, session_id)

                    # Clinical retrieval is the slowest step (1.4s typical, 15s
                    # worst case measured) and the prompt builder only USES it for
                    # moves in MOVES_THAT_USE_CLINICAL_KB — for every other move
                    # the result was fetched and thrown away. Fetch only when
                    # needed, and cap the wait: the guidance prompt has a safe
                    # no-grounding fallback, so a slow retrieval degrades the
                    # grounding rather than the whole reply.
                    _t = time.perf_counter()
                    context = ""
                    if _needs_rag(message, mood, move):
                        try:
                            context = await asyncio.wait_for(
                                rag_service.retrieve_clinical_context(message, mood),
                                timeout=WS_RAG_TIMEOUT_S,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Clinical retrieval exceeded %.1fs | move=%s — continuing without grounding",
                                WS_RAG_TIMEOUT_S, move,
                            )
                    relevant_context = context
                    tm["rag"] = (time.perf_counter() - _t) * 1000

                    # Kept under separate headings: they carry different epistemic
                    # weight and the model must not read the user's own past words
                    # as clinical evidence (same separation voice.py uses).
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

                    _t = time.perf_counter()
                    style_exemplars = await rag_service.retrieve_style_exemplars(
                        move=move,
                        affect_valence=_affect_valence_for_mood(mood),
                        register=_register_for_language(language),
                        k=2,
                    )
                    tm["exemplars"] = (time.perf_counter() - _t) * 1000
                    system_prompt = build_therapeutic_system_prompt(
                        context=merged_context,
                        mood=mood,
                        move=move,
                        language=language,
                        style_exemplars=style_exemplars,
                    )

                    buf = ""
                    _t_llm = time.perf_counter()
                    # therapy_llm_service (generation chain), not sentiment_service
                    # (Groq-first, built for the fast routing call).
                    async for piece in therapy_llm_service.stream_response_for_move(
                        history=history, system_prompt=system_prompt
                    ):
                        if "llm_first_token" not in tm:
                            tm["llm_first_token"] = (time.perf_counter() - _t_llm) * 1000
                        full_response += piece
                        buf += piece
                        sentences, buf = _flush_sentences(buf)
                        for sentence in sentences:
                            if "to_first_sentence_total" not in tm:
                                tm["to_first_sentence_total"] = (time.perf_counter() - t_turn) * 1000
                            await websocket.send_text(
                                json.dumps({"type": "sentence", "content": sentence})
                            )

                    if buf.strip():
                        if "to_first_sentence_total" not in tm:
                            tm["to_first_sentence_total"] = (time.perf_counter() - t_turn) * 1000
                        await websocket.send_text(
                            json.dumps({"type": "sentence", "content": buf.strip()})
                        )
                    tm["llm_stream_total"] = (time.perf_counter() - _t_llm) * 1000

                await websocket.send_text(json.dumps({"type": "done", "risk_level": level}))
                tm["total_to_done"] = (time.perf_counter() - t_turn) * 1000
                logger.info(
                    "Chat WS timing | session=%s | crisis=%s | move=%s | %s",
                    session_id, is_crisis, move,
                    " | ".join(f"{k}={v:.0f}ms" for k, v in tm.items()),
                )

                # Trajectory bookkeeping used to sit BEFORE the reply (it blocked
                # the first sentence); nothing in the reply depends on it, so it
                # now runs after the user already has their answer.
                await _record_trajectory(db, user_id, session_id, mood, risk_score)

                # selected_move is persisted so response-quality problems are
                # diagnosable straight from the DB (voice already does this).
                ai_msg = ChatMessage(
                    session_id=session.id, role="assistant",
                    content=full_response, detected_mood=mood,
                    selected_move=move,
                )
                db.add(ai_msg)
                if level != "LOW":
                    session.risk_level = level
                    db.add(session)
                await db.commit()

            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error("WebSocket turn error | session=%s | %s", session_id, str(e), exc_info=True)
                try:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Processing failed"}))
                except Exception:
                    # Client already disconnected mid-turn — nothing left to notify.
                    raise WebSocketDisconnect(code=1006)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected | session=%s", session_id)
