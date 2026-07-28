# Real-Time Voice Implementation Plan

**Goal:** Make the AI psychologist feel like talking to a real person — hands-free, interruptible, sub-second voice-to-voice.

**Decisions made:** Deepgram for both STT + TTS · full duplex with barge-in · cascaded (not speech-to-speech).

---

## 1. Why cascaded, not speech-to-speech

This is the most important architectural decision, and it's driven by safety, not convenience.

The app's `CLAUDE.md` documents a deliberate **"Hard-Coded Safety Net"**: `CrisisNode` returns deterministic templates — never generative output — for suicide/self-harm. That guarantee requires **inspecting text before anything is spoken**:

```
audio → STT → TEXT → sentiment/risk_score → route (crisis template | therapy) → TTS → audio
                      ▲
                      └── safety checkpoint lives here
```

End-to-end speech-to-speech maps audio directly to audio inside one model. There is no text to inspect, so there is no place to compute `risk_score`, no place to intercept, and no way to force a deterministic crisis response. For a mental-health product, that is not an acceptable trade at any latency.

Supporting reasons (not the deciding ones):
- A well-engineered cascaded pipeline with streaming TTS (100–250ms TTFA) **beats some S2S models** on voice-to-voice latency — cascaded is not the slow option.
- Cascaded costs roughly **10x less at scale**.
- Cascade remains the production default for most voice-agent workloads in 2026.

---

## 2. What's wrong with the current implementation

| Layer | Current | Blocking problem |
|---|---|---|
| STT | Browser Web Speech API | `interimResults = false` → fires only on final result; **unsupported in Firefox/Safari**; click-to-talk only |
| TTS | Browser `SpeechSynthesis` | Robotic, no voice control, inconsistent per-platform — will never feel human |
| Turn-taking | Manual mic button | Real conversation needs automatic speech-end detection |
| Interruption | None | Can't barge in; must wait for the AI to finish |
| Measured latency | **~1.1s to first token** (`tests/test_latency.py`, SSE `/chat/stream`) | Too slow; target is sub-second to first **audio** |

**Also important:** Groq Whisper **cannot** be reused here. It buffers full speech segments before returning results, making it unsuitable for interactive streaming — it's a batch/async transcription engine (164–299x real-time throughput), not a live one.

---

## 3. Target architecture

```
┌─ BROWSER ────────────────────────────────────────────────┐
│  mic → getUserMedia → AudioWorklet                       │
│         └─ downsample to 16kHz mono PCM16                 │
│         └─ send binary frames (~20ms each)                │
│                                                           │
│  playback ← AudioWorklet queue ← binary audio frames      │
│         └─ flush queue instantly on barge-in              │
└───────────────────────┬───────────────────────────────────┘
                        │  ONE WebSocket, binary + JSON mixed
                        ▼
┌─ SERVER: /ws/voice/{session_id}?token=… ─────────────────┐
│                                                           │
│  ┌ inbound audio task ─────────────────────────────────┐ │
│  │  PCM → Deepgram STT (streaming WS)                  │ │
│  │    ├─ interim transcripts → push to UI (live text)  │ │
│  │    ├─ VAD SpeechStarted → if AI talking: BARGE-IN   │ │
│  │    └─ UtteranceEnd → finalize turn ──────────┐      │ │
│  └───────────────────────────────────────────────┼──────┘ │
│                                                  ▼        │
│  ┌ turn pipeline ──────────────────────────────────────┐ │
│  │  asyncio.gather(                                     │ │
│  │     sentiment_service.analyze_sentiment(text),       │ │
│  │     rag_service.retrieve_clinical_context(text)      │ │ ← overlapped
│  │  )                                                   │ │
│  │        │                                             │ │
│  │  risk_score ≥ 8 ? ──yes──→ CRISIS_MESSAGE (fixed)   │ │
│  │        │ no                                          │ │
│  │        ▼                                             │ │
│  │  stream_therapeutic_response() → tokens              │ │
│  │        └─ _flush_sentences() → complete sentences    │ │ ← reuse existing helper
│  │                 │                                    │ │
│  │                 ▼                                    │ │
│  │  Deepgram Aura-2 TTS (streaming WS) → audio frames  │ │
│  │        └─ cancellable on barge-in                    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  Persist: ChatMessage rows + risk_level (same as text)    │
└───────────────────────────────────────────────────────────┘
```

**Reuses what already exists:** `_flush_sentences()` (sentence boundaries), `sentiment_service`, `rag_service`, `_get_or_create_user_session()`, `_load_history()`, `CRISIS_MESSAGE`, and the JWT-via-query-param auth pattern from `/ws/chat/{session_id}`.

---

## 4. Provider choice

| Layer | Provider | Rationale |
|---|---|---|
| **STT** | Deepgram Nova (streaming WS) | True word-by-word streaming, 100–300ms first partial, built-in VAD events (`SpeechStarted` / `UtteranceEnd`) — the VAD comes free, no separate Silero model to host |
| **TTS** | Deepgram Aura-2 (streaming WS) | ~90ms TTFB optimized, **$30/1M chars — half of ElevenLabs Flash ($60/1M)**; same vendor/SDK/key as STT |
| **LLM** | Groq (unchanged) | Already integrated, ~300ms generation |

ElevenLabs Flash is marginally faster (~75ms vs ~90ms) but 2x the cost and a second vendor — a ~15ms difference nobody can perceive.

**Using Deepgram's built-in VAD instead of self-hosted Silero is a deliberate simplification**: it removes a PyTorch model from the request path on a 2-vCPU VM that's already CPU-bound by the BGE embeddings + cross-encoder reranker.

---

## 5. Latency budget

| Stage | Target | Notes |
|---|---|---|
| VAD end-of-speech | ~100ms | Deepgram `UtteranceEnd`, tunable |
| STT final transcript | ~150ms | overlaps with VAD |
| Sentiment (safety gate) | ~150ms | Groq |
| RAG retrieval | **~0ms** | `asyncio.gather`'d with sentiment |
| LLM → first sentence | ~300ms | Groq streaming |
| TTS → first audio | ~90ms | Aura-2 |
| **Voice-to-voice total** | **~800ms** | |

Sub-500ms is reachable with predictive TTS pre-roll and partial-transcript handoff, but **800ms first, then optimize** — don't over-engineer before measuring.

Crisis path is faster (~400ms): skips RAG and LLM generation entirely, TTS starts on a fixed string immediately.

---

## 6. Implementation phases

### Phase V1 — Server voice pipeline
**New:** `app/services/voice_service.py`
- `DeepgramSTTStream`: wraps streaming STT WS; async-iterates `(transcript, is_final, speech_started, utterance_end)` events
- `DeepgramTTSStream`: accepts sentences, yields audio frames; `cancel()` for barge-in
- Both fail soft — if `DEEPGRAM_API_KEY` is absent, voice endpoints 503 but text chat is unaffected

**New:** `app/api/routes/voice.py` → `WS /api/v1/ws/voice/{session_id}?token=…`
- Same JWT-query-param auth + `user_id`-matches-token check as `/ws/chat`
- Two concurrent `asyncio` tasks (inbound audio, outbound audio) coordinated by an `asyncio.Event` for barge-in
- Frame protocol: binary = audio, text = JSON control/events

**Modified:** `app/core/config.py` — `DEEPGRAM_API_KEY`, `DEEPGRAM_STT_MODEL=nova-3`, `DEEPGRAM_TTS_MODEL=aura-2-thalia-en`, `VOICE_ENABLED`
**Modified:** `requirements.txt` — `deepgram-sdk` (pinned, per the lesson learned from the torch/sentence-transformers version drift that hung the VM ingest)
**Modified:** `.env.example` — document the new vars

**Acceptance:** connect, speak, get spoken reply; interrupting mid-reply stops audio within ~200ms; saying "I want to end my life" produces the deterministic crisis template, spoken.

---

### Phase V2 — Browser voice UI
**New:** `app/static/audio-capture.js` (AudioWorklet: mic → 16kHz PCM16 frames)
**New:** `app/static/audio-playback.js` (AudioWorklet: queued playback + instant flush)

**Modified:** `app/static/index.html` — voice-call view: large mic orb, live transcript, state label ("Listening…" / "Thinking…" / "Speaking…"), end-call button
**Modified:** `app/static/style.css` — orb states (idle/listening/thinking/speaking) via the existing CSS-var design system; respects `prefers-reduced-motion`
**Modified:** `app/static/app.js` — voice mode toggle, binary WS handling, barge-in on local speech detection

**UI principle:** during a voice call the interface goes *quieter*, not busier — one breathing orb, minimal text, no chat bubbles competing for attention. Amplitude-reactive animation so the user can see they're being heard.

**Acceptance:** works in Chrome + Edge + Safari (AudioWorklet is universally supported, unlike Web Speech API); mic permission denial degrades to text chat with a clear message.

---

### Phase V3 — Latency validation
**New:** `tests/test_voice_latency.py` — measures, using a real WAV fixture:
- time-to-first-audio-frame (voice-to-voice)
- barge-in stop latency
- crisis-path time-to-first-audio
- reports min/avg/p50/p95/max like the existing `_report()` helper

**Modified:** `tests/test_latency.py` — keep as-is; voice tests live separately since they need audio fixtures

**Acceptance:** p50 voice-to-voice < 1200ms on the Azure D2s_v3, barge-in < 300ms. (Deliberately looser than the 800ms target — the VM is CPU-constrained and shares cores with the reranker.)

---

### Phase V4 — Deploy
- `git push` → pull on VM → `pip install` → restart `psych-api`
- Add `DEEPGRAM_API_KEY` to the VM's `.env`
- **Requires HTTPS**: `getUserMedia` is blocked on insecure origins, so the mic will not work until the DuckDNS domain + certbot TLS step is finished. Voice is gated behind that.
- nginx already passes `Upgrade`/`Connection` headers with `proxy_read_timeout 300s` — no config change needed for the new WS route

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| **HTTPS is a hard prerequisite** | Mic access is impossible without it — finish the domain/TLS step before V4. Voice cannot be demoed over `http://20.98.221.86`. |
| 2-vCPU VM is already CPU-bound | Voice adds no local ML (Deepgram is remote); the reranker stays the bottleneck. Watch for contention under concurrent calls. |
| Deepgram cost overrun | Free-tier credits cover dev. Add a per-session duration cap before any real users. |
| VAD false triggers in noisy rooms | Tune `utterance_end_ms`; keep a manual push-to-talk fallback toggle in the UI. |
| Barge-in race conditions | Single `asyncio.Event` as the one source of truth for "AI is speaking"; cancel TTS task, flush client queue on the same signal. |
| Provider outage | Fail soft: voice endpoints 503, text chat keeps working. Never let voice failure break the core product. |

---

## 8. Open question — one real cost decision

Deepgram needs an API key. Free tier includes credits sufficient for development and testing. Before real users, add a session-duration cap.

**Everything else is decided. Recommended next step: Phase V1.**

---

## Sources

- [Cascaded vs Speech-to-Speech Voice Architecture — Inworld](https://inworld.ai/resources/cascaded-vs-speech-to-speech-voice-architecture)
- [Speech-to-Speech vs Cascade — Deepgram](https://deepgram.com/learn/speech-to-speech-vs-cascade-voice-agent-architecture)
- [Cascaded Voice AI vs Speech-to-Speech: The 2026 Architecture Decision — Future AGI](https://futureagi.com/blog/cascaded-voice-ai-vs-speech-to-speech-2026/)
- [Groq vs OpenAI Whisper: Real Benchmarks (2026)](https://dev.to/howmindswork/groq-vs-openai-whisper-real-benchmarks-for-voice-transcription-2026-46lk)
- [Best STT Providers 2026 — Coval](https://www.coval.ai/blog/best-speech-to-text-providers-in-2026-independent-benchmarks-and-how-to-choose/)
- [Best Text-to-Speech APIs in 2026 — Future AGI](https://futureagi.com/blog/best-text-to-speech-providers-2026/)
- [Deepgram Pricing 2026 — TextToLab](https://texttolab.com/blog/deepgram-pricing)
- [Real-Time TTS with WebSockets — Deepgram Docs](https://developers.deepgram.com/docs/tts-websocket-streaming)
- [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk)
