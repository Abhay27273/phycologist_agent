import json
import asyncio
import re
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
from app.graph.nodes.sentiment import _build_multimodal_hint, _detect_language
from app.graph.nodes.crisis import crisis_message_for

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
            language = _detect_language(payload.message)
            is_crisis = risk_score >= 8
            level = _risk_level(risk_score)

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
            language = _detect_language(payload.message)
            is_crisis = risk_score >= 8
            level = _risk_level(risk_score)

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
                _, session = await _get_or_create_user_session(db, user_id, session_id)
                db.add(ChatMessage(session_id=session.id, role="user", content=message))
                await db.commit()

                multimodal_hint = _build_multimodal_hint(
                    {"audio_features": audio_raw, "video_features": video_raw}
                )
                analysis, history = await asyncio.gather(
                    sentiment_service.analyze_sentiment(message + multimodal_hint),
                    _load_history(db, session_id),
                )
                mood = analysis.get("mood", "neutral")
                risk_score = int(analysis.get("risk_score", 0))
                language = _detect_language(message)
                is_crisis = risk_score >= 8
                level = _risk_level(risk_score)

                await websocket.send_text(
                    json.dumps({"type": "meta", "mood": mood, "risk_score": risk_score})
                )

                full_response = ""

                if is_crisis:
                    full_response = crisis_message_for(language)
                    await websocket.send_text(
                        json.dumps({"type": "sentence", "content": full_response})
                    )
                else:
                    _clinical = {"anxious", "depressed", "lonely", "angry", "stressed",
                                 "fearful", "hopeless", "guilty", "confused"}
                    if len(message.split()) >= 4 or mood in _clinical:
                        context = await rag_service.retrieve_clinical_context(message, mood)
                    else:
                        context = ""

                    history.append({"role": "user", "content": message})

                    buf = ""
                    async for token in sentiment_service.stream_therapeutic_response(
                        history=history, context=context, mood=mood, language=language
                    ):
                        full_response += token
                        buf += token
                        sentences, buf = _flush_sentences(buf)
                        for sentence in sentences:
                            await websocket.send_text(
                                json.dumps({"type": "sentence", "content": sentence})
                            )

                    if buf.strip():
                        await websocket.send_text(
                            json.dumps({"type": "sentence", "content": buf.strip()})
                        )

                await websocket.send_text(json.dumps({"type": "done", "risk_level": level}))

                ai_msg = ChatMessage(
                    session_id=session.id, role="assistant",
                    content=full_response, detected_mood=mood,
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
