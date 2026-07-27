# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Psych-Platform-Core** is an AI-powered psychological support backend using a stateful cognitive architecture (LangGraph) to simulate therapeutic interaction. It serves as the central intelligence for text clients, voice agents, and video avatars.

Key characteristics:
- Python 3.11+ FastAPI backend with async/await throughout
- LangGraph-based orchestration (stateful, cyclical conversation flow — not a linear chatbot)
- Google Gemini 2.5 Flash as primary LLM; Groq (llama-3.3-70b) as free-tier fallback
- PostgreSQL for persistent user/session/message storage
- Dual vector DB support: Pinecone (cloud) or Qdrant (local file-based), toggled via `VECTOR_DB_BACKEND`
- SQLAlchemy ORM with Alembic migrations

**All development commands must be run from `psych-platform-core/`** (the directory containing `app/`, `requirements.txt`, `alembic.ini`, etc.).

## Environment Setup

1. **Virtual Environment** (required):
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Environment Variables** (copy `.env.example` → `.env`):

   Required:
   - `DATABASE_URL`: `postgresql://user:pass@127.0.0.1:5432/psych_db`
   - `GOOGLE_API_KEY`: Google AI Studio key for Gemini

   Vector DB (pick one backend):
   - `VECTOR_DB_BACKEND`: `"pinecone"` (default) or `"qdrant"`
   - `PINECONE_API_KEY` + `PINECONE_INDEX_NAME`: required when backend=pinecone
   - `QDRANT_PATH` + `QDRANT_COLLECTION_NAME`: for local Qdrant (no key needed)

   Optional:
   - `GROQ_API_KEY`: enables Groq fallback LLM
   - `GEMINI_MODEL`: defaults to `gemini-2.5-flash`
   - `CORS_ORIGINS`: comma-separated allowed origins

   On Windows, `localhost` in `DATABASE_URL` is auto-converted to `127.0.0.1` by the config validator.

4. **Database Initialization**:
   ```powershell
   alembic upgrade head
   ```

5. **Build the vector index** (after adding PDFs to `data/`):
   ```powershell
   python scripts/ingest.py
   ```

## Development Commands

### Run the API Server
```powershell
uvicorn app.api.server:app --reload          # development (hot-reload)
uvicorn app.api.server:app --host 0.0.0.0 --port 8000  # production
```
API at `http://127.0.0.1:8000` | Docs at `/api/v1/docs`

### Testing
```powershell
pytest                                        # all tests
pytest tests/test_request.py                 # single file
pytest tests/test_request.py::test_chat_endpoint -v  # specific test
pytest --cov=app tests/                      # with coverage

python tests/test_request.py                 # manual endpoint test (requires running server)
```

### Linting & Formatting
```powershell
black app/
ruff check app/
mypy app/          # optional type checking
```

### Database Migrations
```powershell
alembic revision --autogenerate -m "description"
alembic upgrade head
alembic downgrade -1
```

## Architecture

### The Cognitive Graph (LangGraph)

Defined in `app/graph/workflow.py`. The routing logic is **in `workflow.py` directly** — `edges.py` is a stub.

```
[User Message]
    ↓
[SentimentNode]  → analyzes mood & risk_score (0–10) via Gemini
    ↓
[Conditional Router]  → risk_score >= 8 → Crisis; else → Therapy
    ↙                                          ↘
[CrisisNode]                              [TherapyNode]
(deterministic safety template)           (RAG retrieval + Gemini response)
    ↓                                          ↓
                    [END]
```

**State** (`PsychologicalState` TypedDict in `app/domain/state.py`):
- `messages`: conversation history (append-only via `add` reducer)
- `user_id`, `session_id`: request context
- `current_mood`: detected emotion
- `risk_score`: 0–10 danger assessment
- `is_crisis`: boolean routing flag
- `relevant_context`: RAG-retrieved clinical passages

Graph is invoked with `config={"configurable": {"thread_id": session_id}}` for `MemorySaver`-backed multi-turn persistence.

### Request Flow (POST /api/v1/chat)

1. Pydantic `ChatInput` validation
2. Get-or-create User + ChatSession in DB; persist user message
3. Invoke LangGraph with thread_id = session_id
4. SentimentNode → GeminiService.analyze_sentiment()
5. Router: crisis or therapy path
6. TherapyNode: RAGService.retrieve_clinical_context() → GeminiService.generate_therapeutic_response()
7. Persist assistant response to DB
8. Return `ChatOutput(response, detected_mood, risk_level)`

### RAG Pipeline (`app/services/rag_service.py`)

The retrieval pipeline is sophisticated — not simple vector search:

1. **Query expansion**: Mood-specific keyword injection (e.g., "anxious" → appends "anxiety CBT exposure therapy...")
2. **Dual MMR search**: Runs on both raw and expanded queries (k=12, fetch_k=48)
3. **Noise filtering**: Drops short, low-signal chunks
4. **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
5. **Composite scoring**: Combines semantic, lexical, and topic-based scores
6. Returns top 4 documents with source/page metadata

Embedding model: `BAAI/bge-base-en-v1.5` (HuggingFace, runs locally).

### Service Layer Abstraction

`LLMService` (abstract base in `app/services/llm_interface.py`) exposes:
- `analyze_sentiment(text) → Dict[mood, risk_score]`
- `generate_therapeutic_response(history, mood, context) → str`

Graph nodes depend only on `LLMService`; swap implementations in `build_psychology_graph()` in `workflow.py` without touching nodes.

### Configuration

`app/core/config.py` uses Pydantic `BaseSettings`. Validation runs at startup:
- `force_ip_on_windows()`: Converts `localhost` → `127.0.0.1` in `DATABASE_URL`
- `validate_backend_credentials()`: Enforces `PINECONE_API_KEY` when `VECTOR_DB_BACKEND=pinecone`
- `parse_cors()`: Normalizes `CORS_ORIGINS` to a comma-separated string

### Database Schema

Three SQLAlchemy ORM tables in `app/infrastructure/models.py`:

- **users**: `id` (UUID PK), `email` (unique), `created_at`
- **chat_sessions**: `id` (UUID PK), `user_id` (FK), `risk_level` (LOW/MEDIUM/HIGH), `summary` (long-term memory, nullable), `created_at`
- **chat_messages**: `id` (int PK), `session_id` (FK), `role` (user/assistant), `content`, `detected_mood` (nullable), `timestamp`

## Key Design Patterns

**Hard-Coded Safety Net**: The CrisisNode uses deterministic templates only — never generative LLM output — for suicide/self-harm responses. This is intentional to prevent hallucination in life-safety scenarios.

**Stateful Graph over Linear Chains**: LangGraph enables cyclic flows and emotional state persistence across turns, which is not possible with simple RAG + LLM chains.

**Dual Vector Backend**: `VECTOR_DB_BACKEND=qdrant` requires no cloud credentials and stores data in `./qdrant_data/` locally — useful for development without Pinecone access.

## Windows-Specific Notes

- `app/api/server.py` sets `asyncio.WindowsSelectorEventLoopPolicy()` at startup
- Always use `127.0.0.1` (not `localhost`) in `.env` to avoid asyncpg DNS hang
- The config validator auto-corrects `localhost` as a fallback, but explicit `127.0.0.1` is more reliable

## Adding a New Graph Node

1. Create `app/graph/nodes/my_node.py` as a callable class:
   ```python
   class MyNode:
       def __init__(self, llm_service: LLMService): ...
       async def __call__(self, state: PsychologicalState) -> dict:
           return {"field": value}  # partial state update only
   ```
2. Register in `workflow.py` via `workflow.add_node("name", MyNode(llm_service))`
3. Add edges and conditional routing in `workflow.py`
