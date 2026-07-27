# Psych-Platform-Core — Full Implementation Plan
**Target Review: Codex**  
**Date: 2026-07-01**  
**Current baseline:** FastAPI + LangGraph + Groq + Qdrant (local) · ~1-2 s latency achieved

---

## Executive Summary

Three phases bring the platform from a working prototype to a production-grade system suitable for audio, chat, and video integration:

| Phase | Theme | Outcome |
|-------|-------|---------|
| 1 | Production Baseline | Auth, persistent memory, rate limits, Redis |
| 2 | Brain Improvements | Long-term memory, mood continuity, RAG knowledge expansion |
| 3 | Audio / Video | WebSocket streaming, TTS sentence events, multimodal input |

Each task lists the exact files to create or modify, the packages to install, and the acceptance criterion.

---

## Phase 1 — Production Baseline

### 1.1 JWT Authentication

**Why:** Every endpoint is currently open. JWT secures user identity without a database round-trip per request.

**Packages:**
```
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

**Files to create / modify:**

| File | Action | Notes |
|------|--------|-------|
| `app/core/security.py` | CREATE | `create_access_token()`, `verify_token()`, `hash_password()`, `verify_password()` |
| `app/api/routes/auth.py` | CREATE | `POST /api/v1/auth/register`, `POST /api/v1/auth/login` → returns `{access_token, token_type}` |
| `app/api/dependencies.py` | MODIFY | Add `get_current_user(token: str = Depends(oauth2_scheme)) -> User` |
| `app/api/routes/chat.py` | MODIFY | Add `current_user: User = Depends(get_current_user)` to both endpoints |
| `app/core/config.py` | MODIFY | Add `JWT_SECRET_KEY: str`, `JWT_ALGORITHM: str = "HS256"`, `JWT_EXPIRE_MINUTES: int = 1440` |

**`app/core/security.py` skeleton:**
```python
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)

def create_access_token(sub: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": sub, "exp": expire}, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

def decode_token(token: str) -> str:
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    return payload["sub"]  # user_id
```

**`app/infrastructure/models.py` additions:**
```python
# Add hashed_password column to users table
hashed_password = Column(String, nullable=False)
```
Then generate migration: `alembic revision --autogenerate -m "add_password_to_users"`

**Acceptance:** `POST /api/v1/chat` with no Bearer token returns HTTP 401.

---

### 1.2 Persistent LangGraph Checkpointer (AsyncPostgresSaver)

**Why:** `MemorySaver` holds conversation state in RAM only. A server restart silently drops all multi-turn context. Every user loses their session on deploy.

**Package:**
```
langgraph-checkpoint-postgres==2.0.x   # from langgraph extras
psycopg[async,pool]==3.2.x
```

**Files to modify:**

| File | Change |
|------|--------|
| `app/graph/workflow.py` | Replace `MemorySaver()` with `AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)` |
| `app/core/config.py` | Expose `DATABASE_URL` (already exists); no change needed |
| `requirements.txt` | Add `langgraph-checkpoint-postgres`, `psycopg[async,pool]` |

**`app/graph/workflow.py` target change:**
```python
# BEFORE
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# AFTER
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def build_psychology_graph_async():
    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        await checkpointer.setup()   # creates checkpoint tables if missing
        graph = workflow.compile(checkpointer=checkpointer)
        return graph
```

**Startup hook in `app/api/server.py`:**
```python
@app.on_event("startup")
async def startup():
    global psych_graph
    psych_graph = await build_psychology_graph_async()
```

**Why not SQLite?** `AsyncSqliteSaver` works for single-process dev only — it's incompatible with multiple uvicorn workers. PostgreSQL checkpointer scales horizontally.

**Acceptance:** Kill and restart server mid-conversation; user continues seamlessly.

---

### 1.3 Rate Limiting (slowapi + Redis)

**Why:** Without rate limits, a single malicious client can exhaust Groq API quota and drive up LLM costs.

**Packages:**
```
slowapi==0.1.9
redis[asyncio]==5.0.x
```

**Files to modify:**

| File | Change |
|------|--------|
| `app/api/server.py` | Mount `Limiter` from slowapi, add exception handler |
| `app/api/routes/chat.py` | Add `@limiter.limit("20/minute")` to chat endpoint |
| `app/core/config.py` | Add `REDIS_URL: str = "redis://127.0.0.1:6379/0"` |

**`app/api/server.py` additions:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, storage_uri=settings.REDIS_URL)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

**Default limits (configurable via env):**
- `/api/v1/chat`: 20 requests/minute per IP
- `/api/v1/chat/stream`: 10 requests/minute per IP  
- `/api/v1/auth/login`: 5 requests/minute per IP (brute-force protection)

**Acceptance:** 21st request in a minute returns HTTP 429 with `Retry-After` header.

---

### 1.4 Redis RAG Cache (replace in-process dict)

**Why:** The current `_RAG_CACHE` dict in `rag_service.py` is per-process. With multiple uvicorn workers (production), each worker builds its own cache — wasted CPU. Redis gives a single shared cache.

**Files to modify:**

| File | Change |
|------|--------|
| `app/services/rag_service.py` | Replace `_RAG_CACHE` dict with Redis `aioredis` client |
| `app/api/server.py` | Initialize Redis pool on startup, inject into RAGService |
| `app/core/config.py` | `REDIS_URL` (already added in 1.3) |

**`app/services/rag_service.py` target pattern:**
```python
import redis.asyncio as aioredis
import pickle

class RAGService:
    def __init__(self, redis_client=None):
        ...
        self._redis = redis_client   # None → fall back to in-process dict
        self._local_cache: dict = {} # fallback when Redis unavailable

    async def _cache_get(self, key: str):
        if self._redis:
            val = await self._redis.get(f"rag:{key}")
            return pickle.loads(val) if val else None
        entry = self._local_cache.get(key)
        if entry and (time.monotonic() - entry[0]) < _RAG_CACHE_TTL:
            return entry[1]
        return None

    async def _cache_set(self, key: str, value: str):
        if self._redis:
            await self._redis.setex(f"rag:{key}", _RAG_CACHE_TTL, pickle.dumps(value))
        else:
            self._local_cache[key] = (time.monotonic(), value)
```

**Acceptance:** Two workers serving the same query: second worker hits Redis, cross-encoder skipped for both.

---

### 1.5 Qdrant Server Mode (Docker, multi-worker ready)

**Why:** Local file-based Qdrant (`QdrantClient(path=...)`) uses a SQLite lock — only ONE process can open it at a time. Multi-worker uvicorn (`--workers 4`) will crash with `portalocker.AlreadyLocked`.

**Docker command:**
```bash
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

**Files to modify:**

| File | Change |
|------|--------|
| `app/services/rag_service.py` | `QdrantClient(url="http://127.0.0.1:6333")` when `QDRANT_MODE=server` |
| `app/core/config.py` | Add `QDRANT_MODE: str = "local"` (`"local"` or `"server"`), `QDRANT_URL: str = "http://127.0.0.1:6333"` |
| `.env.example` | Document `QDRANT_MODE`, `QDRANT_URL` |

**`rag_service.py` branch:**
```python
if settings.QDRANT_MODE == "server":
    client = QdrantClient(url=settings.QDRANT_URL)
else:
    client = QdrantClient(path=qdrant_path)  # existing local path
```

**Acceptance:** `uvicorn app.api.server:app --workers 4` starts without lock errors.

---

## Phase 2 — Brain Improvements

### 2.1 SummaryNode — Long-Term Memory Compression

**Why:** LangGraph's `MemorySaver` / `AsyncPostgresSaver` appends every message to the state. After ~30 turns the token count in LLM calls grows linearly, increasing latency and cost. Summarization compresses old turns into one paragraph.

**Trigger:** After every `SUMMARY_EVERY_N_TURNS = 10` assistant messages.

**Files to create / modify:**

| File | Action |
|------|--------|
| `app/graph/nodes/summary.py` | CREATE — `SummaryNode` class |
| `app/domain/state.py` | ADD `session_summary: Optional[str]`, `turn_count: int` to `PsychologicalState` |
| `app/graph/workflow.py` | ADD `SummaryNode` after `therapeutic_response`; conditional edge: summarize every N turns |
| `app/api/routes/chat.py` | After graph invocation, persist `state["session_summary"]` to `chat_sessions.summary` |
| `app/infrastructure/models.py` | `summary` column already exists — no schema change needed |

**`app/graph/nodes/summary.py` skeleton:**
```python
SUMMARY_EVERY_N = 10
_SUMMARY_PROMPT = """
Summarize this therapy conversation in 3-5 sentences. Capture:
- The user's core concerns and emotional themes
- Key coping strategies discussed
- Progress or setbacks noted
- Current risk level

Conversation:
{conversation}

Previous summary (incorporate if present):
{prior_summary}
"""

class SummaryNode:
    def __init__(self, llm_service: LLMService): ...
    
    async def __call__(self, state: PsychologicalState) -> dict:
        turn_count = state.get("turn_count", 0) + 1
        if turn_count % SUMMARY_EVERY_N != 0:
            return {"turn_count": turn_count}
        
        recent_messages = state["messages"][-SUMMARY_EVERY_N * 2:]
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages)
        prior = state.get("session_summary") or ""
        
        prompt = _SUMMARY_PROMPT.format(conversation=conversation, prior_summary=prior)
        summary = await self.llm_service.complete(prompt)
        
        # Trim message history — keep last 4 messages + summary instead of full log
        trimmed = state["messages"][-4:]
        return {
            "session_summary": summary,
            "messages": trimmed,   # replace, not append (use operator.replace reducer)
            "turn_count": turn_count,
        }
```

**State change for `messages` trimming:**
```python
# In state.py — add a second messages field or use Annotated with replace reducer
# Simplest: keep messages as append-only, SummaryNode writes to a separate
# trimmed_messages field used as the LLM context window
```

**Graph routing:**
```python
def should_summarize(state):
    return "summarize" if (state.get("turn_count", 0) + 1) % SUMMARY_EVERY_N == 0 else END

workflow.add_conditional_edges("therapeutic_response", should_summarize,
    {"summarize": "summarize_node", END: END})
workflow.add_edge("summarize_node", END)
```

**Acceptance:** After 10 turns, `chat_sessions.summary` is populated; subsequent LLM calls include the summary instead of raw history.

---

### 2.2 Longitudinal Mood Awareness

**Why:** A user returning for their 5th session is still treated as new. Loading prior risk level + summary seeds the graph with emotional continuity.

**Files to modify:**

| File | Change |
|------|--------|
| `app/domain/state.py` | ADD `prior_session_summaries: List[str]`, `longitudinal_risk: str` |
| `app/api/routes/chat.py` | Before `psych_graph.ainvoke()`, query DB for last 3 sessions → inject into initial state |
| `app/infrastructure/database.py` | ADD `get_recent_sessions(user_id, limit=3) -> List[ChatSession]` |

**Injection point in `chat.py`:**
```python
recent_sessions = await get_recent_sessions(db, user_id=payload.user_id, limit=3)
prior_summaries = [s.summary for s in recent_sessions if s.summary]
longitudinal_risk = recent_sessions[0].risk_level if recent_sessions else "LOW"

initial_state = {
    "messages": [{"role": "user", "content": payload.message}],
    "user_id": payload.user_id,
    "session_id": payload.session_id,
    "prior_session_summaries": prior_summaries,
    "longitudinal_risk": longitudinal_risk,
    # ... existing fields
}
```

**SentimentNode usage of longitudinal context:**
```python
# In sentiment prompt, add:
prior_context = ""
if state.get("prior_session_summaries"):
    prior_context = "Prior session summaries:\n" + "\n---\n".join(state["prior_session_summaries"][:3])
```

**Acceptance:** Risk-score jump detection — if prior `longitudinal_risk = HIGH` and new score is 3, system flags possible suppression rather than treating as low-risk.

---

### 2.3 RAG Knowledge Base Expansion

#### 2.3.1 Existing Datasets (already cloned — need ingestion scripts)

| Dataset | Location | Format | Rows | Ingestion Strategy |
|---------|----------|--------|------|--------------------|
| **CounselChat** | `datasets/counselchat/counsel-chat/data/counselchat-data.csv` | CSV (`questionText`, `answerText`, `topic`) | ~2,800 Q&A pairs | Convert each Q+A to a chunk; use `topic` column as metadata |
| **EmpatheticDialogues** | `datasets/empathetic_dialogues/empatheticdialogues/` | CSV (`situation`, `emotion`, `utterance`) | ~24,850 conversations | Convert situation+utterance to passage chunks; tag `emotion` as metadata |
| **ECC** (Emotion Cause Corpus) | `datasets/ecc/ECC/ECC_ALL/ECC_train_ALL.jsonl` | JSONL | ~5,000 utterances | Extract `utterance` + `emotion` + `cause` as context passages |
| **PsyDial** | `datasets/pysdial/PsyDial/` | Check README for format | ~2,000 dialogues | Extract multi-turn dialogues as passages |

**Script to create:** `scripts/ingest_datasets.py`

```python
"""
Converts datasets to structured text chunks and upserts into Qdrant.
Run after ingest.py (PDF ingestion).
"""
import csv, json, pathlib
from scripts.ingest import get_vector_store, chunk_text  # reuse existing helpers

def ingest_counselchat():
    rows = csv.DictReader(open("datasets/counselchat/counsel-chat/data/counselchat-data.csv"))
    for row in rows:
        text = f"Question: {row['questionText']}\n\nTherapist Response: {row['answerText']}"
        chunk_and_upsert(text, metadata={
            "source": "CounselChat",
            "topics": row.get("topic", ""),
            "type": "qa_pair"
        })

def ingest_empathetic_dialogues():
    for split in ["train", "valid", "test"]:
        rows = csv.DictReader(open(f"datasets/empathetic_dialogues/empatheticdialogues/{split}.csv"))
        for row in rows:
            text = f"Situation: {row['situation']}\nResponse: {row['utterance']}"
            chunk_and_upsert(text, metadata={
                "source": "EmpatheticDialogues",
                "topics": row.get("context", ""),
                "emotion": row.get("emotion", "")
            })
```

#### 2.3.2 New PDFs to Download and Add to `data/`

These are open-access or public domain:

| Source | Title | Topics | Access |
|--------|-------|--------|--------|
| **NIMH** | Depression — What You Need to Know (2024) | depression, treatment | nimh.nih.gov/health/publications — free download |
| **NIMH** | Anxiety Disorders (2024) | GAD, panic, social anxiety | nimh.nih.gov/health/publications — free download |
| **NIMH** | Understanding PTSD and PTSD Treatment | trauma, EMDR, CPT | nimh.nih.gov/health/publications — free download |
| **SAMHSA** | Trauma-Informed Care in Behavioral Health Services (TIP 57) | trauma, crisis | store.samhsa.gov — free PDF, 276 pages |
| **SAMHSA** | Substance Abuse Treatment for Persons with Co-Occurring Disorders (TIP 42) | dual diagnosis | store.samhsa.gov — free PDF |
| **CANMAT 2023** | Clinical Practice Guidelines for MDD | depression, treatment algorithms | Published in Can J Psychiatry; free PMC versions |
| **Beck Institute** | CBT Worksheet Compendium (public) | CBT, thought records, behavioral activation | beckinstitute.org — free resources |
| **University of Michigan DBT** | DBT Skills Training Manual (Linehan excerpts) | DBT, emotion regulation, distress tolerance | Partial public domain excerpts |
| **ACT Mindfulness** | Acceptance and Commitment Therapy — Introduction | ACT, values, defusion | Multiple university course materials, CC-licensed |
| **IAPT Handouts** | Low Intensity CBT Interventions | CBT, behavioral activation, worry | NHS IAPT materials — publicly available |

**Download script:** `scripts/download_clinical_pdfs.sh` (see Appendix A)

#### 2.3.3 HuggingFace Dataset Downloads

```bash
# Install huggingface_hub CLI
pip install huggingface_hub

# MentalChat16K — 9,775 mental health conversations with clinical rationale
huggingface-cli download ShenLab/MentalChat16K --repo-type dataset --local-dir datasets/mentalchat16k

# Amod counseling conversations — 3,512 detailed therapy exchanges
huggingface-cli download Amod/mental_health_counseling_conversations --repo-type dataset --local-dir datasets/amod_counseling

# SmileChat — multitask mental health support dialogues (Chinese + English translated)
huggingface-cli download maxwellyin/SMILE --repo-type dataset --local-dir datasets/smile

# HOPE corpus — 212 therapy sessions with adherence labels
huggingface-cli download nbertagnolli/counsel-chat --repo-type dataset --local-dir datasets/hope_corpus
```

**Ingest priority (highest clinical value first):**
1. MentalChat16K — has `instruction` + `output` + `input` columns with CBT rationale
2. Amod counseling — long-form therapist responses with context
3. CounselChat (already present) — real therapist Q&A with topic labels
4. EmpatheticDialogues (already present) — 32 emotion labels

#### 2.3.4 Enhanced RAG Metadata Schema

Current chunks have `source` and `page` only. Expand to:

```python
metadata = {
    "source": "CANMAT_2023.pdf",        # filename
    "page": 14,                          # page number
    "topics": "depression,MDD",          # comma-separated topic tags (feeds _MOOD_TOPIC_HINTS)
    "technique": "behavioral_activation", # CBT/DBT/ACT technique label
    "disorder": "MDD",                   # primary disorder
    "evidence_level": "A",               # CANMAT evidence levels A/B/C/D
    "chunk_type": "clinical_guideline",  # "guideline"|"qa_pair"|"dialogue"|"worksheet"
}
```

**Update `_MOOD_TOPIC_HINTS` in `rag_service.py`** to use new metadata fields:
```python
_MOOD_TOPIC_HINTS = {
    "anxious":   {"anxiety", "GAD", "panic", "phobia"},
    "depressed": {"depression", "MDD", "crisis", "behavioral_activation"},
    "lonely":    {"relationship", "interpersonal", "attachment"},
    "angry":     {"DBT", "emotion_regulation", "impulse_control"},
    "stressed":  {"stress", "mindfulness", "coping"},
    "fearful":   {"anxiety", "exposure", "phobia"},
    "hopeless":  {"depression", "crisis", "safety_plan"},
    "guilty":    {"CBT", "shame", "self_compassion"},
    "confused":  {"grounding", "dissociation", "DBT"},
    "traumatized": {"PTSD", "trauma", "EMDR", "CPT"},   # NEW
    "grieving":    {"grief", "loss", "bereavement"},     # NEW
}
```

#### 2.3.5 Updated Ingest Pipeline (`scripts/ingest.py`)

Key improvements:
- Accept `--source-dir` CLI arg to ingest from multiple directories
- Read sidecar `.meta.json` files for pre-assigned metadata
- Deduplicate chunks by content hash before upsert
- Progress bar with `tqdm`
- Parallel PDF loading with `concurrent.futures.ThreadPoolExecutor`

---

### 2.4 Sentiment Node Enhancement — Structured Output

**Why:** Current `analyze_sentiment()` returns a dict parsed from LLM text. This can fail silently if Groq returns unexpected format.

**Change:** Use Pydantic structured output via `instructor` library.

**Package:** `instructor==1.x`

**Files to modify:**

| File | Change |
|------|--------|
| `app/services/groq_service.py` | Add `analyze_sentiment_structured()` using instructor |
| `app/graph/nodes/sentiment.py` | Use structured method |

```python
# app/services/groq_service.py addition
import instructor
from pydantic import BaseModel, Field

class SentimentResult(BaseModel):
    mood: str = Field(..., description="Primary emotion: anxious/depressed/lonely/angry/stressed/fearful/hopeless/guilty/confused/neutral")
    risk_score: int = Field(..., ge=0, le=10, description="Risk score 0-10")
    rationale: str = Field(..., description="One-sentence rationale for risk score")
    detected_themes: list[str] = Field(default_factory=list)

client_structured = instructor.from_groq(Groq(api_key=settings.GROQ_API_KEY))

async def analyze_sentiment_structured(self, text: str) -> SentimentResult:
    return await asyncio.to_thread(
        client_structured.chat.completions.create,
        model=self.model,
        response_model=SentimentResult,
        messages=[{"role": "user", "content": f"Analyze: {text}"}],
    )
```

**Acceptance:** Malformed LLM output raises `ValidationError`, not silent dict parse failure.

---

## Phase 3 — Audio / Video Integration

### 3.1 WebSocket Endpoint for Bidirectional Audio

**Why:** HTTP SSE is unidirectional (server → client). Audio/video pipelines need the client to stream audio chunks continuously while receiving responses. WebSocket supports both directions on one connection.

**Architecture:**
```
Client (Web/Mobile)
    |
    | ws://host/api/v1/ws/{session_id}
    |
    ↓ Binary frames: raw PCM audio chunks (16kHz, 16-bit mono)
    ↑ Text frames: JSON {"type": "transcript"|"token"|"done"|"error", "data": "..."}
```

**File to create:** `app/api/routes/websocket.py`

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio, json

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200   # collect 200ms of audio before VAD check

@router.websocket("/ws/{session_id}")
async def audio_ws(websocket: WebSocket, session_id: str,
                   db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                audio_buffer.extend(data["bytes"])
                # Check if we have enough for VAD
                if len(audio_buffer) >= SAMPLE_RATE * 2 * (CHUNK_DURATION_MS / 1000):
                    if _vad_detects_speech_end(audio_buffer):
                        transcript = await _transcribe(bytes(audio_buffer))
                        audio_buffer.clear()
                        
                        await websocket.send_json({"type": "transcript", "data": transcript})
                        
                        # Stream therapy response
                        async for token in _stream_response(transcript, session_id, db):
                            await websocket.send_json({"type": "token", "data": token})
                        await websocket.send_json({"type": "done"})
            
            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        pass
```

**VAD integration (optional, improves UX):**
```
silero-vad==4.0.x   # lightweight torch-based VAD, works on CPU
```

**STT integration (pluggable, no vendor lock-in):**

| Option | Latency | Cost | Notes |
|--------|---------|------|-------|
| Groq Whisper (`whisper-large-v3-turbo`) | ~200ms | $0.04/hr audio | Fastest option, already have Groq key |
| OpenAI Whisper API | ~300ms | $0.006/min | Reliable fallback |
| Local `faster-whisper` (int8) | ~500ms CPU | Free | `ctranslate2` based |

**Recommended:** Groq Whisper — reuses existing Groq client.

```python
# In groq_service.py
async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
    response = await asyncio.to_thread(
        self._client.audio.transcriptions.create,
        model="whisper-large-v3-turbo",
        file=("audio.wav", audio_bytes, "audio/wav"),
        language=language,
    )
    return response.text
```

**Acceptance:** Client streams 3s of speech audio; server returns transcript within 500ms, then streams therapy tokens.

---

### 3.2 Sentence Boundary SSE Events for TTS / Avatar Sync

**Why:** TTS engines work best with complete sentences. Streaming tokens one-by-one causes choppy audio. SSE should buffer tokens until a sentence boundary (`.`, `!`, `?`, `...`) then emit a `sentence` event so TTS can begin speaking while the LLM generates the rest.

**Files to modify:**

| File | Change |
|------|--------|
| `app/api/routes/chat.py` | Add `StreamingFormat` enum (`"tokens"` / `"sentences"`); add `sentence` SSE event type |
| `app/domain/state.py` | Add `stream_format: str` to `ChatInput` |

**Sentence boundary emitter:**
```python
import re

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=\.\.\.)\s*')

async def sentence_stream(token_gen):
    """
    Wraps an async token generator.
    Yields (event_type, data) tuples:
      - ("token", token_text)     — always emitted for real-time display
      - ("sentence", sentence)    — emitted when a complete sentence is ready for TTS
    """
    buf = ""
    async for token in token_gen:
        buf += token
        yield "token", token
        
        parts = _SENTENCE_END.split(buf, maxsplit=1)
        if len(parts) > 1:
            yield "sentence", parts[0].strip()
            buf = parts[1]
    
    if buf.strip():
        yield "sentence", buf.strip()   # flush last fragment
```

**SSE event format for TTS clients:**
```
event: token
data: {"text": " feeling", "index": 42}

event: sentence
data: {"text": "It sounds like you're feeling really overwhelmed right now.", "sentence_index": 3}

event: done
data: {"total_tokens": 87}
```

**Acceptance:** TTS client receives first `sentence` event within 300ms of first token.

---

### 3.3 Multimodal Input Extension

**Why:** Video avatar clients can detect facial expressions; audio clients have pitch/energy features. Injecting these into the therapy prompt improves response quality (e.g., "I can see you're speaking quickly and your voice sounds tense").

**Files to modify:**

| File | Change |
|------|--------|
| `app/domain/state.py` | Extend `ChatInput` with optional multimodal fields |
| `app/graph/nodes/sentiment.py` | Incorporate multimodal signals into risk/mood analysis |
| `app/graph/nodes/therapy.py` | Pass multimodal context to LLM prompt |

**Extended `ChatInput`:**
```python
class AudioFeatures(BaseModel):
    speech_rate_wpm: Optional[float] = None      # words per minute
    pitch_mean_hz: Optional[float] = None        # mean fundamental frequency
    pitch_variance: Optional[float] = None       # pitch instability → anxiety signal
    energy_db: Optional[float] = None            # loudness → agitation signal
    pause_ratio: Optional[float] = None          # ratio of silence → depression signal
    voice_tremor: Optional[bool] = None          # detected tremor → fear/anxiety

class VideoFeatures(BaseModel):
    dominant_emotion: Optional[str] = None       # from facial AU analysis (e.g., "sad")
    eye_contact_ratio: Optional[float] = None    # 0-1, low → depression/shame
    facial_action_units: Optional[list[str]] = None  # AU codes e.g. ["AU1", "AU4", "AU15"]
    gaze_direction: Optional[str] = None         # "down", "away", "direct"

class ChatInput(BaseModel):
    user_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    audio_features: Optional[AudioFeatures] = None   # NEW
    video_features: Optional[VideoFeatures] = None   # NEW
    stream_format: Optional[str] = "tokens"           # NEW: "tokens" | "sentences"
```

**Sentiment node prompt injection:**
```python
# Build multimodal context string
modal_context = ""
if state.get("audio_features"):
    af = state["audio_features"]
    modal_context += f"\nAudio signals: speech_rate={af.get('speech_rate_wpm')} wpm, "
    modal_context += f"pitch_variance={af.get('pitch_variance')}, tremor={af.get('voice_tremor')}"
if state.get("video_features"):
    vf = state["video_features"]
    modal_context += f"\nVideo signals: dominant_emotion={vf.get('dominant_emotion')}, "
    modal_context += f"eye_contact={vf.get('eye_contact_ratio')}"
```

**Add to `PsychologicalState`:**
```python
audio_features: Optional[dict]
video_features: Optional[dict]
```

**Acceptance:** ChatInput with `audio_features.voice_tremor=true` causes sentiment node to increase risk awareness for anxiety; visible in response tone.

---

## Phase 4 — Observability & Reliability (Production Hardening)

### 4.1 Structured Logging

**Package:** `structlog==24.x`

Replace ad-hoc `logger.info()` with structured JSON logs:
```python
import structlog
log = structlog.get_logger()
log.info("rag.retrieved", session_id=session_id, mood=mood, doc_count=len(docs), latency_ms=elapsed)
```

**Key log events to instrument:**
- `graph.invoked` — session_id, user_id
- `sentiment.analyzed` — mood, risk_score, latency_ms
- `rag.retrieved` — mood, doc_count, cache_hit, latency_ms
- `llm.generated` — model, token_count, latency_ms
- `crisis.triggered` — session_id, risk_score (always log, never sample)

### 4.2 Health & Metrics Endpoint

Extend `app/api/routes/health.py`:
```
GET /api/v1/health          → {"status": "up", "db": "up", "qdrant": "up", "groq": "up"}
GET /api/v1/metrics         → Prometheus text format (fastapi-prometheus)
```

### 4.3 Database Connection Pooling

Current `database.py` uses `create_async_engine` with default pool size. For production:
```python
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,      # validates connections before use
)
```

### 4.4 Graceful Shutdown

```python
@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()       # close DB pool
    await redis_client.aclose()  # close Redis pool
    # Qdrant gRPC client closes automatically
```

---

## Appendix A — Clinical PDF Download Script

Save as `scripts/download_clinical_pdfs.sh`:

```bash
#!/usr/bin/env bash
# Downloads open-access clinical psychology resources into data/
set -e
mkdir -p data

# NIMH Booklets (public domain, US Government)
curl -o "data/nimh_depression_2024.pdf" \
  "https://www.nimh.nih.gov/sites/default/files/health/publications/depression/depression_booklet.pdf"

curl -o "data/nimh_anxiety_disorders_2024.pdf" \
  "https://www.nimh.nih.gov/sites/default/files/health/publications/anxiety-disorders/anxiety-disorders-booklet.pdf"

curl -o "data/nimh_ptsd_treatment.pdf" \
  "https://www.nimh.nih.gov/sites/default/files/health/publications/post-traumatic-stress-disorder-ptsd/ptsd.pdf"

# SAMHSA TIP 57 — Trauma-Informed Care (public domain)
curl -o "data/samhsa_tip57_trauma_informed_care.pdf" \
  "https://store.samhsa.gov/sites/default/files/sma14-4816.pdf"

echo "Download complete. Run: python scripts/ingest.py"
```

---

## Appendix B — Requirements Additions

Add to `psych-platform-core/requirements.txt`:

```
# Phase 1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
slowapi==0.1.9
redis[asyncio]==5.0.8
langgraph-checkpoint-postgres==2.0.14
psycopg[async,pool]==3.2.3

# Phase 2
instructor==1.6.4
tqdm==4.67.1
huggingface_hub==0.28.1

# Phase 3
silero-vad==4.0.1        # optional VAD

# Phase 4
structlog==24.4.0
prometheus-fastapi-instrumentator==7.0.0
```

---

## Appendix C — Migration Checklist

In order — each step is a prerequisite for the next:

- [ ] **1.1** JWT auth + `hashed_password` column + migration
- [ ] **1.2** `langgraph-checkpoint-postgres` + AsyncPostgresSaver startup hook
- [ ] **1.3** slowapi rate limiting + Redis connection
- [ ] **1.4** Redis RAG cache (replace in-process dict)
- [ ] **1.5** Qdrant Docker server mode
- [ ] **2.1** `PsychologicalState` extended + `SummaryNode` + graph edge
- [ ] **2.2** `get_recent_sessions()` + longitudinal state injection
- [ ] **2.3** Run `scripts/download_clinical_pdfs.sh` + `ingest_datasets.py` + re-index Qdrant
- [ ] **2.4** `instructor` structured sentiment output
- [ ] **3.1** WebSocket endpoint + Groq Whisper transcription
- [ ] **3.2** Sentence boundary SSE emitter
- [ ] **3.3** `ChatInput` multimodal extension
- [ ] **4.x** Structlog, metrics, DB pool, graceful shutdown

---

## Latency Budget (target after all phases)

| Step | Current | Target | How |
|------|---------|--------|-----|
| Sentiment analysis | ~150ms | ~120ms | Groq structured output (no regex parse) |
| RAG retrieval | ~300ms (cached), ~800ms (cold) | ~100ms (Redis hit), ~600ms (cold) | Redis shared cache |
| LLM generation (first token) | ~300ms | ~250ms | Groq streaming, no thinking mode |
| Total to first token (warm) | ~600ms | ~400ms | Parallel RAG + structured sentiment |
| Total to first TTS sentence | ~900ms | ~650ms | Sentence boundary buffering |
| WebSocket audio (speech → first token) | N/A | ~700ms | Groq Whisper (~200ms) + sentiment + RAG |
