# Making a Chatbot Behave Like a Psychologist — RAG, Chat & Latency

**Project:** Psych-Platform-Core · **Scope:** text chat pipeline, clinical retrieval, latency engineering

---

## 1. The design problem

A generic LLM wrapper produces *advice*. A psychologist produces *containment first, then
one small step*. The gap is not model quality — it's architecture. Three properties had to
be engineered rather than prompted:

1. **Groundedness** — coping suggestions must trace to real clinical literature, not model priors.
2. **Emotional state awareness** — the reply must depend on *how* the user is doing, and on
   what they said in previous sessions, not just the last message.
3. **A safety floor that cannot hallucinate** — for suicide/self-harm, generative output is
   unacceptable at any quality level.

These map onto three subsystems: a precision-first RAG pipeline, a stateful cognitive graph,
and a deterministic crisis path.

---

## 2. The clinical knowledge base

Source corpus is real clinical literature, not scraped content:

| Source | Type |
|---|---|
| *Psychology 2e* (OpenStax, ~88 MB) | textbook |
| *Principles of Psychology* | textbook |
| NICE — *Depression in adults: treatment and management* | guideline |
| PHQ-9 | instrument |
| plus `guidelines/`, `instruments/`, `taxonomy/` directories | mixed |

**Ingestion** (`scripts/ingest.py`) is deliberately opinionated, because PDF textbooks are
mostly *not* useful therapeutic text:

- **Semantic chunking by default** — `SemanticChunker` (percentile, threshold 85) splits on
  concept boundaries rather than character counts, so a chunk contains a complete idea.
  Falls back to `RecursiveCharacterTextSplitter` (512 / 100 overlap) if unavailable.
- **Page-level rejection** before chunking: pages under 180 chars, and pages detected as
  psychometric forms (checklist density, markers like `"not difficult at all"`, `"0 1 2 3"`,
  `"Total:"`) are dropped. A PHQ-9 answer grid retrieves well and helps a user not at all.
- **Metadata enrichment**: each chunk gets an inferred `source_type`
  (guideline / instrument / taxonomy / textbook) and `topics`
  (anxiety, depression, relationship, stress, crisis) via keyword inference — later used as
  a retrieval-scoring signal.
- Embeddings: **BAAI/bge-base-en-v1.5**, run locally on CPU, normalized. Batched 100 at a time.

Result: ~7,000 chunks (per the tuning note in `rag_service.py`). Vector store is pluggable —
Pinecone (cloud) or Qdrant (local/Docker) behind one factory, since both expose an identical
`max_marginal_relevance_search` signature.

---

## 3. Retrieval: precision over recall

`RAGService.retrieve_clinical_context(query, mood)` is a six-stage pipeline. The guiding
principle is that **three excellent passages beat ten mediocre ones** — irrelevant context
actively degrades a therapeutic reply by inviting generic advice.

1. **Mood-conditioned query expansion.** The detected mood selects a clinical vocabulary
   expansion — `anxious` → `"anxiety CBT exposure therapy cognitive restructuring panic
   disorder"`. This bridges the vocabulary gap between how users talk ("I can't stop
   worrying") and how literature is written.
2. **Dual parallel MMR search.** Two Maximal-Marginal-Relevance searches (`k=3`,
   `fetch_k=12`) run concurrently — one on the raw query, one on the expanded query — via
   `asyncio.gather` + `asyncio.to_thread`. MMR (not plain top-k) enforces diversity so we
   don't return three paraphrases of one paragraph. Raw query preserves the user's actual
   concern; expanded query reaches the clinical framing.
3. **Dedup + noise filtering.** Near-duplicates removed by normalized content. Chunks are
   dropped if under 120 chars, if alphabetic ratio < 0.6 (OCR noise, tables), or if they
   carry known low-signal markers.
4. **Hybrid scoring.** `score = cross_encoder + (0.25 × lexical_overlap) + topic_bonus`.
   The lexical term stabilizes domain terminology; `topic_bonus` (+0.12) rewards chunks
   whose ingest-time topics match the current mood. Cross-encoder reranking
   (`ms-marco-MiniLM-L-6-v2`) is **opt-in via `ENABLE_RERANKER`, default off** — it costs
   1–3 s per request on CPU, which is unaffordable in a conversational loop.
5. **Adaptive top-k.** Rather than a fixed cutoff, keep candidates within `0.30` of the best
   score (floor `0.15`), capped at 3. A weak-match query returns *less* context instead of
   padding with noise.
6. **Source anchoring + caching.** Each passage is prefixed `[source: file.pdf, page: N]` so
   generation stays grounded and auditable. Results cache on `md5(query|mood)` with a 300 s
   TTL, in Redis when available (shared across workers) or an in-process dict.

---

## 4. The cognitive graph — where "psychologist" happens

`app/graph/workflow.py` compiles a LangGraph `StateGraph`, not a linear chain. State
(`PsychologicalState`) carries `messages` (append-only via an `add` reducer), `current_mood`,
`risk_score`, `is_crisis`, `relevant_context`, `session_summary`, and `longitudinal_context`.

```
User message
     ↓
SentimentNode ──→ mood + risk_score (0–10)
     ↓
 risk ≥ 8 ?
   ├── yes → CrisisNode      (deterministic template — no LLM)
   └── no  → TherapyNode     (RAG + grounded generation)
                  ↓
        every 10 messages → SummaryNode → END
```

**Why a graph:** risk assessment must *gate* generation, not run alongside it. Routing on
inspected text is the safety property; a linear chain cannot express it.

**The therapeutic contract** (`therapeutic_prompt.py`) encodes clinical conversation
technique as hard constraints — this is what most changes the felt quality of replies:

- Reflect the feeling in specific language and validate *why it makes sense* — **first**.
- **Do not rush to advice, diagnosis, or reassurance**; help the user feel heard first.
- **At most one** small coping step, and only if supported by retrieved context.
- If context is weak, **acknowledge uncertainty instead of inventing evidence**.
- Exactly **one** gentle open-ended question, to hand control back to the user.
- 3–5 short sentences; no jargon, no bullet lists, no scripted phrases.

The "validate before advise" ordering and the one-question rule are what separate this from a
tips generator.

**Memory operates at two timescales.** Within a session, the last 10 messages plus a
LangGraph checkpointer (`AsyncSqliteSaver` dev / `AsyncPostgresSaver` prod) preserve
continuity across restarts. Across sessions, `SummaryNode` compresses every 10 messages into
a 2–3 sentence summary persisted to `chat_sessions.summary`; the last 3 prior summaries are
injected as `[RISK] summary → [RISK] summary`, giving the model **mood trajectory** — the
difference between a stranger and someone who remembers you.

**Multimodal hooks** exist for voice/avatar clients: optional audio features (tone, speech
rate, energy) and video features (dominant emotion, gaze avoidance, FACS action units) are
rendered into a plain-English hint appended to the sentiment prompt, so *how* something was
said can shift risk scoring.

**The safety floor.** `CrisisNode` returns a fixed string — never generative output. It names
that it is an AI and directs to emergency services and a hotline. Risk bands: `≥8` HIGH,
`≥4` MEDIUM, else LOW; MEDIUM/HIGH is persisted to the session.

---

## 5. Latency engineering

Four transports, each for a different consumer:

| Endpoint | Granularity | Consumer |
|---|---|---|
| `POST /chat` | full reply | simple clients |
| `POST /chat/stream` | per token (SSE) | text UI |
| `POST /chat/stream/sentences` | per sentence (SSE) | TTS / avatar |
| `WS /ws/chat/{id}` | per sentence | live voice agents |

Sentence-granularity streaming exists because TTS needs complete sentences — server-side
buffering via a sentence-boundary regex means clients don't reimplement it.

**What actually bought the time:**

- **Model choice per task.** Gemini 2.5 Flash was rejected for the hot path: its thinking
  phase buffers all tokens (~8–10 s) even in blocking mode. Groq `llama-3.3-70b-versatile`
  streams a first token in ~100 ms. Sentiment classification was later split onto
  `llama-3.1-8b-instant` — measured 120–250 ms vs 320–500 ms warm, with **no accuracy loss at
  the crisis threshold** (verified against a set of crisis/non-crisis messages before shipping).
- **Parallelism.** Sentiment analysis and history load run under one `asyncio.gather`; the two
  MMR searches run concurrently. Blocking CPU work (embedding, reranking) is pushed to
  threads so the event loop keeps serving.
- **Conditional RAG.** Messages under 4 words with a non-clinical mood skip retrieval
  entirely — saves 1.5–4 s on "hi there".
- **Reranker off by default.** The single largest latency win; hybrid lexical + topic scoring
  recovers most of the precision.
- **Early `meta` event.** Mood and risk are emitted before generation starts, so the UI can
  react (risk badge, orb state) during the LLM wait.
- **Startup warm-up.** A throwaway sentiment call at boot absorbs the one-time client
  connection cost (~800–1000 ms) that would otherwise hit the first real user.
- **Cache key excludes user/session** — `md5(query|mood)` — so common phrasings warm globally.

**Test methodology** (`tests/test_latency.py`) hits a live server with **no mocking** — real
LLM calls, real retrieval — because the point is production-path latency. It reports
min/avg/p50/p95/max per scenario and asserts on **median** (robust to provider hiccups).
Ceilings: blocking 8 s, crisis 3 s, first token 4 s, first sentence 6 s. Coverage includes
trivial vs clinical vs crisis paths, cold-vs-warm cache, and 3-way concurrency.

One methodological lesson is documented in the suite itself: ~25 real LLM calls in ~90 s can
exceed a free-tier per-minute quota, and the resulting SDK retry-backoff adds 4–5 s that
looks exactly like an application regression. `INTER_SAMPLE_DELAY_S = 2.0` paces the suite,
and the docstring records how to distinguish throttling from a real regression.

---

## 6. Honest limitations

- Not clinically validated; no licensed-practitioner review of output quality.
- Crisis routing depends on an LLM risk score. The *response* is deterministic, but the
  *decision to use it* is not — a missed classification is the main residual risk.
- Cross-encoder reranking is disabled by default, trading some retrieval precision for latency.
- Single-VM deployment; no load testing beyond light concurrency.
- Retrieval quality is bounded by the corpus — a few textbooks and guidelines, not a
  comprehensive clinical library.
