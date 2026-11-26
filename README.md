# Psych-Platform-Core

> **AI-powered psychological support backend** — a stateful cognitive architecture that listens, detects risk, retrieves clinical evidence, and responds like a responsible, culturally-aware mental-health support companion.

---

## What This Actually Is (Plain English)

Strip away the architecture diagrams below and here's the short version: this is a backend that powers an AI you can **text, call, or video-call**, built specifically for mental-health conversations rather than general chat.

What makes it different from "a chatbot wired up to an LLM":

- **It remembers the conversation, not just the last message.** Every turn updates a persistent state (current mood, risk level, which therapeutic technique was just used) that carries forward across the whole session — and a compressed summary carries forward across *separate* sessions, so returning users aren't starting from zero.
- **It decides *how* to respond, not just *what* to say.** Before generating a reply, it picks a specific counseling technique (reflect, validate, ask an open question, gently challenge a distorted thought, etc.) the way a trained counselor would vary their approach turn to turn — not a chatbot's one-size-fits-all "helpful assistant" voice.
- **It looks things up before answering.** Rather than trusting an LLM's own training data on clinical topics, it retrieves real passages from clinical textbooks and guidelines and grounds its response in them.
- **If things look dangerous, a human-written safety script takes over — not the AI.** When a message scores high enough on risk, the system stops generating text altogether and returns a fixed, pre-approved crisis message with real helpline numbers. No model ever gets a chance to improvise in a life-safety moment.
- **One backend, three front doors.** The exact same decision-making pipeline serves a text chat widget, a real-time voice call, and a video call with a reactive avatar — the interface changes, the "brain" behind it doesn't.

The sections below are the engineering detail behind each of those five bullets.

---

## Why We Built This

Mental health support in India faces two compounding crises: **acute therapist scarcity** (fewer than 1 mental-health professional per 100,000 people) and **deep stigma** that keeps people from seeking care at all. Many people's first disclosure of distress happens not in a clinic, but in a private message, a late-night search, or a voice call with no one on the other end.

We built Psych-Platform-Core to be a **responsible first-contact layer** — not a replacement for licensed care, but a system that can:

1. **Listen before it speaks** — most chatbots reply immediately with advice or coping techniques; real clinicians reflect and validate first.
2. **Detect genuine crisis signals** — including passive suicidal ideation, South Asian somatic idioms ("sar bhaari hai", "neend nahi aati"), and Hinglish distress phrasing that standard English-first models routinely under-score.
3. **Respond in the user's own language and register** — English, Hindi (Devanagari), or Hinglish (Latin-script Hindi), with culturally native phrasing, not translated clinical English.
4. **Never hallucinate a crisis response** — the safety path is 100% deterministic code, no LLM interpolation.
5. **Power text, voice, and avatar interfaces from one backend** — the same cognitive graph drives a web chat UI, a real-time voice call (Deepgram/Sarvam STT + TTS with barge-in), and a future video-avatar agent.

The system is explicit about its limits: it is not a psychologist, it does not diagnose, and it immediately escalates any crisis-level signal to real human services (Tele-MANAS 14416, 112).

---

## High-Level Architecture

```mermaid
flowchart TD
    subgraph Clients
        A["Web Chat UI\n(app.js / index.html)"]
        B["Voice Call Client\n(voice-call.js)"]
        C["Video / Avatar Client\n(video-call.js + face-tracking.js)"]
    end

    subgraph FastAPI["FastAPI Backend (psych-platform-core)"]
        direction TB
        R1["POST /api/v1/chat\n(blocking)"]
        R2["GET  /api/v1/chat/stream\n(SSE tokens)"]
        R3["WS   /api/v1/ws/chat/{session_id}\n(persistent WebSocket)"]
        R4["WS   /api/v1/ws/voice/{session_id}\n(real-time speech)"]
        R5["Auth / Memory / Feedback\nroutes"]
    end

    subgraph CognitiveGraph["LangGraph Cognitive Graph"]
        direction TB
        SN["SentimentNode\n(mood · risk 0-10 · language · distortion)"]
        STR["StrategyNode\n(therapeutic move selection)"]
        TH["TherapyNode\n(RAG + LLM response)"]
        CR["CrisisNode\n(deterministic safety template)"]
        SUM["SummaryNode\n(rolling session compression)"]

        SN -->|"risk ≥ 8"| CR
        SN -->|"risk < 8"| STR
        STR --> TH
        TH -->|"every N turns"| SUM
        TH --> DONE["END"]
        CR --> DONE
        SUM --> DONE
    end

    subgraph Services["AI & Data Services"]
        LLM["LLM Waterfall\nNebius Nemotron → OpenRouter → Gemini → Groq\n(circuit-breaker per 429)"]
        RAG["RAGService\nclinical_kb · style_exemplars · patient_memory"]
        MEM["MemoryService\nfacts · trajectory · longitudinal context"]
        VOICE["Voice Pipeline\nDeepgram STT/TTS (EN) · Sarvam (HI/Hinglish)\nSmart Turn v3 · barge-in"]
    end

    subgraph Storage
        PG["PostgreSQL\nusers · sessions · messages\nmood_trajectory · user_facts\nLangGraph checkpointer"]
        VDB["Vector DB\nQdrant (local) or Pinecone (cloud)\npsych-brain · psych-style · psych-patient-memory"]
        RC["Redis (optional)\nRAG result cache · rate limiting"]
    end

    A & B & C --> FastAPI
    FastAPI --> CognitiveGraph
    CognitiveGraph --> Services
    Services --> Storage
```

---

## The Cognitive Graph — Five Nodes

The core intelligence is built with **LangGraph** — a framework for structuring an AI's decision-making as a flowchart of steps (nodes) with conditional branches, rather than one big prompt that tries to do everything at once. Each user message walks through up to five of these steps before a reply comes back.

Concretely, this is a **LangGraph StateGraph** compiled with a persistent *checkpointer* — a component that saves the graph's state (mood, risk score, move history, etc.) to a database after every step (SQLite for dev, PostgreSQL for prod). That's what gives every session true multi-turn memory: the graph can resume exactly where a conversation left off, even across separate API requests.

### 1 · SentimentNode
Analyzes each user message and returns:
- **`current_mood`** — one of 16 moods including South Asian idiom categories (`somatic`, `tension`, `anhedonic`)
- **`risk_score`** — 0–10 (passive hopelessness → passive SI → active ideation → immediate crisis)
- **`detected_language`** — `en` | `hi` | `hinglish` (heuristic, no network call)
- **`cognitive_distortion_detected`** — triggers the `reality_test` move downstream

The scoring prompt is carefully calibrated for Hinglish active-vs-passive suicidal ideation, scoring phrases like "khatam karna chahta hoon" at 8–10 (active desire to end life) rather than discounting them as mere hopelessness.

### 2 · StrategyNode
Selects a **therapeutic move** for this turn, based on:
- Risk band (low 0–3 / mid 4–7)
- Cognitive distortion flag (mandates `reality_test` at weight 50)
- Last 3 moves (no consecutive repetition)
- Turn index (suppresses bare `open_question` on opening turns — avoids interrogation)
- Silent-move constraint (no two consecutive turns with nothing for the user to reply to)

**Move taxonomy:** `simple_reflection`, `complex_reflection`, `affirmation`, `open_question`, `summarise_and_check`, `normalise`, `psychoeducation`, `sit_with_it`, `reality_test`

The `sit_with_it` move (~18–20% baseline from AnnoMI clinician data) is the move most absent from LLM bots and most characteristic of skilled human clinicians.

### 3 · TherapyNode
Generates the response using:
- **RAGService** — RAG (Retrieval-Augmented Generation) means the system searches a knowledge base for relevant real text *before* asking the LLM to respond, and feeds that text in as grounding context — instead of trusting the model's own training data on clinical topics. Here, that's clinical KB (knowledge base) retrieval, skipped for moves that don't need it, saving 1.5–4s
- **Style exemplars** — real AnnoMI therapist turns retrieved by metadata filter (`move + valence + register`), never by semantic similarity to the user's message, so no one's specific situation leaks into another's response
- **Language-native system prompt** — English or Hindi/Hinglish register with separate banned-phrase lists, cultural preamble (family as resource not boundary), and anti-AI-meta-commentary rules
- **Anti-echo guard** — regenerates if response content-word overlap with the user message exceeds 35%

### 4 · CrisisNode
Hard-coded, **LLM-free** safety response for `risk_score ≥ 8`. Returns a bilingual fixed template directing the user to Tele-MANAS (14416), 112, and affirming they are not alone. This is intentional — hallucination in a life-safety response is unacceptable.

### 5 · SummaryNode
Every `N` turns, compresses the conversation into a rolling prose summary persisted to `chat_sessions.summary`. The summary is loaded as `longitudinal_context` in future sessions, giving the assistant continuity across separate conversations.

---

## RAG Pipeline — Clinical Knowledge Base

Before the LLM writes a single word, the user's message is turned into a search query against a vector database of clinical text — so the response can be grounded in actual guidelines and therapy material, not just whatever the model happens to "remember." The steps below are what happens between "user hits send" and "relevant passages land in the prompt":

```
User message
    │
    ├─ Mood-specific query expansion  ("anxious" → "anxiety CBT exposure therapy...")
    ├─ South Asian idiom expansion    ("neend nahi" → "insomnia sleep disturbance depression")
    │
    ├─── MMR search (raw query)   ─┐
    │                               ├─ asyncio.gather (parallel)
    └─── MMR search (expanded)    ─┘
         k=3, fetch_k=12 each
         (MMR = Maximal Marginal Relevance: a search strategy that pulls back
         results balancing "most relevant" against "not all saying the same
         thing," so the top 3 docs aren't 3 near-duplicate passages)
              │
              ▼
    Merge + dedup + noise filter
    (drop: <120 chars, <60% alpha, forms, noisy OCR)
              │
              ▼
    Hybrid scoring:
      Cross-encoder score  (opt-in, ENABLE_RERANKER=true)
    + 0.25 × lexical overlap
    + 0.12 topic bonus (mood ↔ doc metadata)
              │
              ▼
    Top ≤3 docs → prompt [Clinical evidence] block
              │
              ▼
    MD5(query|mood) cache → Redis or in-process dict (TTL 300s)
```

**Knowledge base sources:** clinical psychology textbooks, NICE depression guidelines, PHQ-9, AnnoMI therapy transcripts, CounselChat Q&A, EmpatheticDialogues, PsyDial, ECC datasets.

**Embedding model:** `BAAI/bge-base-en-v1.5` (runs locally on CPU)
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (opt-in)
**Vector backends:** Qdrant (local file, dev) | Pinecone (cloud, prod)

---

## Voice Pipeline

The hard problem in a real-time voice call isn't speech-to-text itself — it's knowing *when the user has actually finished talking* versus just pausing mid-thought, and handling "barge-in" (the user interrupting the AI mid-reply) without it feeling broken. The pipeline below layers three different turn-taking signals (an LLM-based check, a dedicated local ML model called Smart Turn v3, and a plain silence timeout) specifically because no single signal is reliable enough alone.

```
Mic audio (PCM 16-bit mono @ 16kHz)
       │
       ▼
  STT Stream  ───────────── language ────────────────────────────────────
  EN: Deepgram nova-3                                                    │
  HI/Hinglish: Sarvam saaras:v3                                         │
       │                                                                 │
       ├── speech_final → Semantic check (Groq is_utterance_complete)    │
       │                 + Smart Turn v3 ONNX (12–65ms local CPU)        │
       │                                                                 │
       ├── UtteranceEnd (~1000ms silence fallback)                       │
       │                                                                 │
       └── Barge-in: energy duck → word confirm → cancel in-flight turn  │
                                                                         │
Transcribed text → SentimentNode → StrategyNode → RAG → LLM            │
                                                        │                │
                                                        ▼                │
                                               Streaming tokens          │
                                               _flush_sentences()        │
                                                        │                │
                                               TTS Stream  ◄─────────────┘
                                               EN: Deepgram aura-2-thalia-en
                                               HI: Sarvam bulbul:v1 (meera voice)
                                                        │
                                               PCM audio → WebSocket binary frames
```

**Key voice behaviors:**
- **Backchannel ack** ("Mm, I hear you." / "Haan, batao.") spoken immediately after mood is known, masking RAG + LLM latency (~1–3s)
- **Stall watchdog** fires after 2.5s if all turn-detectors fail — guarantees a reply
- **Transparent STT reconnect** on Deepgram socket drop (rate-limited to one attempt per 2s)
- **Barge-in** gated on transcribed words, not raw VAD energy, to prevent noise-triggered interruption

---

## Memory Architecture — Four Layers

| Layer | Store | Description |
|---|---|---|
| **Turn history** | PostgreSQL `chat_messages` | Every user + assistant message, with `detected_mood` and `selected_move` per assistant turn |
| **Session summary** | `chat_sessions.summary` | Rolling prose compressed by SummaryNode every N turns; loaded as `longitudinal_context` |
| **Structured facts** | `user_facts` table | LLM-extracted entities (triggers, coping strategies, core beliefs, goals) with temporal validity; never overwritten, soft-invalidated |
| **Semantic recall** | Qdrant `psych-patient-memory` | Dialogue highlights stored by `store_patient_memory()`, recalled by `retrieve_patient_memory()` with hard `user_id` filter |

**Recall gating** — memory context is injected only when a trigger fires:
1. User explicitly references the past ("last time", "pehle", "yaad hai")
2. Entity overlap — a stored person/event label appears in the current message
3. Long absence ≥ 7 days since last session
4. Trajectory slope — `risk_score` rising > 2.0 over last 3 sessions

> *"A system that opens every reply with a callback to a prior session is not attentive, it's performing attentiveness."* — Memory design principle §3.3

**DPDP compliance:** DPDP (India's Digital Personal Data Protection Act) gives users a right to erasure — so `MemoryService.forget(user_id, fact_id)` is a first-class hard-delete primitive, not a soft flag. "Delete my data" has to actually delete it, not just hide it from future queries.

---

## LLM Routing — Circuit-Breaker Waterfall

In plain terms: the system doesn't depend on any single AI provider. It tries a preferred provider first, and if that provider is rate-limited or down, it automatically falls through to the next one in line — a "circuit breaker" remembers which providers just failed and skips straight past them for the next 65 seconds instead of wasting time retrying a provider that's still down.

```
Generation (therapy responses):
  Nebius Nemotron-120B → OpenRouter (Llama-3.3-70B :free) → Gemini 2.5 Flash → Groq Llama-3.3-70B
  Each provider skipped for 65s automatically after a 429 — no wasted retries.

Sentiment (risk routing — latency-sensitive):
  Groq Llama-3.1-8B-Instant → OpenRouter → Gemini 2.5 Flash
  Groq is first because its ~150ms latency vs ~800ms+ for Gemini kept
  the voice "meta" event from blocking for 10–40s during rate-limit events.

Multiple Groq keys: round-robin load balancing multiplies the per-minute
token budget by N (each free-tier key has its own ~12K tok/min cap).
```

---

## Database Schema

```
users                    chat_sessions             chat_messages
─────────────────────    ──────────────────────    ──────────────────────────
id (UUID PK)             id (UUID PK)              id (int PK)
email (unique)           user_id (FK → users)      session_id (FK → sessions)
hashed_password          created_at                role (user | assistant)
google_sub               risk_level (LOW/MED/HIGH) content
created_at               summary (longitudinal)    detected_mood
                         session_language          selected_move
                                                   timestamp

mood_trajectory          user_facts                session_ratings
──────────────────────   ──────────────────────    ──────────────────────────
user_id, session_id      user_id                   WAI-SR (12-item)
turn_index               fact_text                 SRS (4-item)
mood, risk_score         entity_label              therapeutic alliance score
recorded_at              category, confidence
                         validity_start/end         dependency_signals
                                                   session_count
                                                   night_time_turn_share
                                                   exclusive_reliance_count
                                                   human_support_mention_count
```

> **WAI-SR** (Working Alliance Inventory – Short Revised) and **SRS** (Session Rating Scale) are validated questionnaires used in real therapy research to measure how strong the working relationship between client and therapist feels — reused here as a way to actually measure whether the AI is building a helpful rapport, not just assuming it is.

---

## Project Structure

```
phycologist_system/
├── psych-platform-core/          # Main Python backend
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py       # JWT + Google Sign-In
│   │   │   │   ├── chat.py       # Blocking, SSE, WebSocket chat
│   │   │   │   ├── voice.py      # Real-time voice WebSocket
│   │   │   │   ├── memory.py     # Facts, trajectory, recall
│   │   │   │   └── feedback.py   # WAI-SR / SRS ratings
│   │   │   ├── server.py         # FastAPI app + lifespan warmup
│   │   │   └── limiter.py        # Rate limiting (slowapi)
│   │   ├── core/
│   │   │   ├── config.py         # Pydantic settings + validators
│   │   │   ├── security.py       # JWT helpers
│   │   │   └── exceptions.py
│   │   ├── domain/
│   │   │   └── state.py          # PsychologicalState TypedDict + Pydantic I/O models
│   │   ├── graph/
│   │   │   ├── workflow.py       # LangGraph build + LLM waterfall routing
│   │   │   └── nodes/
│   │   │       ├── sentiment.py  # SentimentNode (mood, risk, language, distortion)
│   │   │       ├── strategy.py   # StrategyNode (therapeutic move selection)
│   │   │       ├── therapy.py    # TherapyNode (RAG + LLM response)
│   │   │       ├── crisis.py     # CrisisNode (deterministic safety)
│   │   │       └── summary.py    # SummaryNode (session compression)
│   │   ├── services/
│   │   │   ├── rag_service.py          # Clinical KB + style + patient memory
│   │   │   ├── therapeutic_prompt.py   # Move instructions (EN + HI) + system prompt builder
│   │   │   ├── memory_service.py       # Facts, trajectory, recall gating, DPDP forget
│   │   │   ├── voice_service.py        # Deepgram STT/TTS stream wrappers
│   │   │   ├── sarvam_voice_service.py # Sarvam STT/TTS (HI/Hinglish)
│   │   │   ├── turn_detector.py        # Smart Turn v3 ONNX (audio-based)
│   │   │   ├── gemini_service.py       # Google Gemini LLM adapter
│   │   │   ├── groq_service.py         # Groq LLM adapter + sentiment
│   │   │   ├── openrouter_service.py   # OpenRouter adapter
│   │   │   ├── nebius_service.py       # Nebius Nemotron adapter
│   │   │   ├── fallback_llm_service.py # Circuit-breaker waterfall
│   │   │   └── round_robin_llm_service.py # Multi-key Groq load balancer
│   │   ├── infrastructure/
│   │   │   ├── models.py         # SQLAlchemy ORM tables
│   │   │   └── database.py       # Async engine + session factory
│   │   └── static/               # Bundled web UI (chat + voice + video)
│   ├── data/                     # Clinical PDFs + Hinglish session transcripts
│   ├── datasets/                 # AnnoMI, CounselChat, PsyDial, ECC, EmpatheticDialogues
│   ├── migrations/               # Alembic DB migrations
│   ├── scripts/                  # ingest.py (build vector index)
│   └── tests/
├── avatar_capture/               # Live Link Face capture pipeline (future)
└── docs/
    └── RAG_CHAT_LATENCY_WRITEUP.md
```

---

## Setup & Configuration

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ (or SQLite for local dev)
- Node.js (optional, for frontend tooling)

### 1 · Environment
```powershell
cd psych-platform-core
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2 · Environment Variables
Copy `.env.example` → `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | `postgresql://user:pass@127.0.0.1:5432/psych_db` or `sqlite+aiosqlite:///./psych_db.sqlite` |
| `GOOGLE_API_KEY` | ✅ | Google AI Studio key for Gemini |
| `VECTOR_DB_BACKEND` | ✅ | `qdrant` (local dev) or `pinecone` (prod) |
| `PINECONE_API_KEY` | if Pinecone | Pinecone cloud key |
| `QDRANT_PATH` | if Qdrant | `./qdrant_data` (local file path) |
| `GROQ_API_KEY` | recommended | Low-latency sentiment + fallback generation |
| `GROQ_API_KEY_2/3/4` | optional | Additional keys for round-robin load balancing |
| `OPENROUTER_API_KEY` | optional | Free-tier aggregated models |
| `NEBIUS_API_KEY` | optional | NVIDIA Nemotron primary generation tier |
| `DEEPGRAM_API_KEY` | for voice | Streaming STT + TTS |
| `SARVAM_API_KEY` | for Hindi voice | Native Hinglish/Hindi STT + TTS |
| `JWT_SECRET_KEY` | prod | Change from default in any non-local env |
| `GOOGLE_OAUTH_CLIENT_ID` | optional | Enables "Sign in with Google" button |
| `REDIS_URL` | optional | Shared RAG cache; falls back to in-process dict |

> **Windows note:** Always use `127.0.0.1` (not `localhost`) in `DATABASE_URL`. The config validator auto-corrects this, but explicit is safer.

### 3 · Database Initialization
```powershell
alembic upgrade head
```

### 4 · Build Vector Index
```powershell
python scripts/ingest.py          # Ingest clinical PDFs → psych-brain collection
# (Optional) python scripts/ingest_annomi.py   # Ingest AnnoMI style exemplars → psych-style
```

### 5 · Run
```powershell
# Development (hot-reload)
uvicorn app.api.server:app --reload

# Production
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

API: `http://127.0.0.1:8000` | Docs: `/api/v1/docs`

---

## API Reference (key endpoints)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create account (email + password) |
| `POST` | `/api/v1/auth/login` | Get JWT token |
| `POST` | `/api/v1/auth/google` | Google ID-token exchange |
| `POST` | `/api/v1/chat` | Blocking chat (returns full response JSON) |
| `GET` | `/api/v1/chat/stream` | SSE token stream |
| `GET` | `/api/v1/chat/stream/sentences` | SSE sentence stream (voice/avatar clients) |
| `WS` | `/api/v1/ws/chat/{session_id}` | Persistent WebSocket text chat |
| `WS` | `/api/v1/ws/voice/{session_id}` | Real-time speech-to-speech call |
| `GET` | `/api/v1/memory/facts` | Get user's stored facts |
| `DELETE` | `/api/v1/memory/facts/{id}` | Hard-delete a fact (DPDP erasure) |
| `GET` | `/api/v1/memory/trajectory` | Mood/risk time series |
| `POST` | `/api/v1/feedback/session` | Submit WAI-SR / SRS rating |
| `GET` | `/health` | Server health check |

**Chat request / response:**
```json
// POST /api/v1/chat
{
  "user_id": "user_abc",
  "session_id": "session_xyz",
  "message": "mujhe bahut udaas feel ho raha hai, neend bhi nahi aati",
  "audio_features": {"tone": "flat", "speech_rate": "slow", "energy": 0.2},
  "video_features": {"dominant_emotion": "sad", "gaze_avoidance": true}
}

// Response
{
  "response": "Itni neend nahi aati aur dil bhi bhaari hai — yeh dono ek saath bahut thaka dete hain. Kab se aisa chal raha hai?",
  "detected_mood": "depressed",
  "risk_level": "MEDIUM"
}
```

---

## Development Commands

```powershell
# Tests
pytest                                                     # all tests
pytest tests/test_request.py -v                           # verbose
pytest --cov=app tests/                                   # with coverage

# Lint / format
black app/
ruff check app/
mypy app/                                                  # optional

# DB migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
alembic downgrade -1
```

---

## Safety Design Principles

1. **Deterministic crisis path** — `CrisisNode` uses zero LLM calls. The safety message is a constant in source code, versioned in git, reviewed like any other change.

2. **Risk scoring calibrated for South Asian phrasing** — The sentiment prompt explicitly instructs the LLM to score "khatam karna chahta hoon" at 8–10 (active ideation), not 6–7 (hopelessness), and to distinguish passive suicidal ideation (PHQ-9 item 9 category) from general fatigue-with-life.

3. **No diagnostic labelling** — The system prompt bans volunteering DSM/ICD category names (depression, anxiety disorder, bipolar) under any circumstance, including hedged forms.

4. **Family as resource** — The cultural preamble prohibits boundary-setting advice about family unless the user explicitly requests it. Indian family contexts are treated as a primary coping resource by default.

5. **Anti-dependency design** — The system tracks exclusive-reliance phrases, night-time session concentration, declining human-support mentions, and session frequency in the `dependency_signals` table. The prompt explicitly forbids claiming exclusive closeness or constant availability.

6. **Privacy-first memory** — Style exemplars are retrieved by move metadata only, never by semantic similarity to the user's message. Patient memory is hard-filtered by `user_id` — leaking one user's context into another's response is treated as a privacy violation, not an accuracy trade-off.

---

## Datasets Used

| Dataset | Usage |
|---|---|
| **AnnoMI** | Expert-annotated MI therapy transcripts; source for style exemplars (`psych-style` collection) |
| **CounselChat** | Real therapist Q&A pairs; clinical KB |
| **PsyDial** | Psychological dialogue dataset |
| **EmpatheticDialogues** | Empathy-annotated conversations |
| **ECC (Emotional Causal Chain)** | Emotion-cause pairs |
| **NICE Depression Guidelines** | Clinical evidence for RAG |
| **PHQ-9** | Screening instrument for severity calibration |
| **OpenStax Psychology 2e** | General psychology textbook for KB |
| **Hinglish sessions** | Self-authored bilingual therapy session transcripts for Hinglish style exemplars |

---

## Roadmap / Phases

- [x] **Phase 1** — LangGraph cognitive core (SentimentNode → crisis / TherapyNode)
- [x] **Phase 2** — Longitudinal memory (SummaryNode, session summary injection)
- [x] **Phase 3** — Multimodal signals (audio features, video/facial action units)
- [x] **Phase 4** — Structured memory (UserFact, MoodTrajectory, MemoryService)
- [x] **Phase 5.1** — Therapeutic alliance measurement (WAI-SR, SRS)
- [x] **Phase 5.2** — Dependency monitoring (DependencySignal table)
- [x] **Phase V** — Real-time voice (Deepgram STT/TTS, Smart Turn v3, barge-in)
- [x] **Hinglish voice** — Sarvam STT/TTS for native Hindi/Hinglish audio
- [ ] **Avatar integration** — Live Link Face capture pipeline → video-avatar client
- [ ] **Fine-tuned sentiment model** — Replace LLM-based risk scoring with a local classifier trained on the calibrated prompt outputs

---

## License

This project is a research and development prototype. The clinical datasets it references carry their own licenses — see each dataset's README before redistribution. No warranty is expressed or implied for any use in production clinical settings.

---

*Built in India. For India. And for anyone who needs to feel heard at 2 AM.*
