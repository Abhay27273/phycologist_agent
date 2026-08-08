import os
import asyncio
import hashlib
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
# Must stay at module level: forces PyTorch/BLAS to load *before* LangGraph's
# C extensions, preventing a KMP_DUPLICATE_LIB_OK access-violation on Windows.
from sentence_transformers import CrossEncoder
from app.core.config import settings

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_MOOD_EXPANSION = {
    "anxious":   "anxiety CBT exposure therapy cognitive restructuring panic disorder",
    "depressed": "depression behavioral activation negative cognition mood disorder",
    "lonely":    "loneliness attachment interpersonal therapy social isolation",
    "angry":     "anger management emotion regulation DBT impulse control",
    "stressed":  "stress coping strategies resilience mindfulness relaxation",
    "fearful":   "phobia fear avoidance graded exposure desensitization",
    "hopeless":  "hopelessness suicidal ideation safety plan crisis intervention",
    "guilty":    "guilt shame cognitive distortion self-compassion CBT",
    "confused":  "dissociation identity emotional dysregulation grounding techniques",
    # Indian idiom moods
    "somatic":   "somatic symptom disorder medically unexplained symptoms depression somatic somatisation bodily distress",
    "tension":   "anxiety GAD stress rumination worry chronic stress tension headache",
    "anhedonic": "anhedonia depression amotivation reward processing low mood",
    "grieving":  "grief bereavement loss complicated grief",
    "traumatized": "trauma PTSD stress response safety stabilization",
}

_MOOD_TOPIC_HINTS = {
    "anxious": {"anxiety", "stress"},
    "depressed": {"depression", "crisis"},
    "lonely": {"relationship"},
    "angry": {"relationship", "stress"},
    "stressed": {"stress", "anxiety"},
    "fearful": {"anxiety"},
    "hopeless": {"depression", "crisis"},
    "guilty": {"depression", "stress"},
    "confused": {"stress"},
    "somatic": {"anxiety", "stress", "depression"},
    "tension": {"anxiety", "stress"},
    "anhedonic": {"depression"},
    "grieving": {"depression", "relationship"},
    "traumatized": {"crisis", "stress"},
}

# South Asian / Hinglish distress idioms → clinical topic expansions.
# Matched against the raw user message text to supplement mood-based expansion.
# Keys are lowercase phrases; values are clinical search terms to append.
_IDIOM_EXPANSIONS: dict[str, set[str]] = {
    "tension":          {"anxiety", "GAD", "stress", "worry", "rumination"},
    "ghabrahat":        {"panic", "palpitations", "acute anxiety"},
    "bechaini":         {"restlessness", "agitation"},
    "dil bhaari":       {"depression", "grief", "low mood"},
    "dil nahi lagta":   {"anhedonia", "depression", "amotivation"},
    "sar bhaari":       {"somatic", "tension headache", "stress"},
    "kamzori":          {"fatigue", "somatic", "depression"},
    "neend nahi":       {"insomnia", "sleep disturbance", "depression", "anxiety"},
    "neend nahi aati":  {"insomnia", "sleep disturbance", "depression", "anxiety"},
    "mann nahi lagta":  {"anhedonia", "amotivation", "depression"},
    "thaka hua":        {"burnout", "fatigue", "depression"},
    "thak gaya":        {"burnout", "fatigue", "depression"},
    "ghabrana":         {"anxiety", "panic", "fear"},
    "udaas":            {"depression", "sadness", "low mood"},
    "bahut udaas":      {"depression", "grief"},
    "akela":            {"loneliness", "isolation", "depression"},
    "bilkul akela":     {"loneliness", "isolation", "depression"},
    "tang aa gaya":     {"hopelessness", "burnout", "existential distress"},
    "tang aa gayi":     {"hopelessness", "burnout", "existential distress"},
    "kuch nahi chahiye": {"depression", "hopelessness", "anhedonia"},
    "rone ka mann":     {"depression", "grief", "low mood"},
    "rona aa raha":     {"depression", "grief"},
    "bahut takleef":    {"distress", "pain", "depression"},
    "bahut dard":       {"grief", "depression", "pain"},
    "kya fayda":        {"hopelessness", "depression", "existential distress"},
    "zindagi se thak":  {"hopelessness", "suicidal ideation", "burnout"},
}

_DEFAULT_EXPANSION = "therapeutic intervention psychology CBT coping skills"

# Retrieval and selection tuning knobs.
# 6/24 → 3/12: cross-encoder is O(n) on CPU; halving candidates roughly halves
# its inference time (~0.4-0.8s vs ~1.2-2.5s) without measurable recall loss
# at this KB size (6,999 chunks). Cap cross-encoder input at 4 docs.
_MMR_K = 3
_MMR_FETCH_K = 12
_MAX_FINAL_DOCS = 3
_MAX_RERANK_DOCS = 4  # hard cap on cross-encoder input regardless of dedup output

_RAG_CACHE_TTL = 300  # seconds


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{3,}", (text or "").lower()))


def _is_low_signal_chunk(text: str) -> bool:
    """Filter chunks that commonly hurt precision (forms, sparse fragments, noisy OCR)."""
    if not text:
        return True

    compact = " ".join(text.split())
    if len(compact) < 120:
        return True

    alpha_chars = sum(ch.isalpha() for ch in compact)
    if alpha_chars / max(len(compact), 1) < 0.6:
        return True

    lowered = compact.lower()
    noise_markers = [
        "copyright",
        "add columns",
        "total:",
        "not difficult at all",
        "healthcare professional",
    ]
    if any(marker in lowered for marker in noise_markers):
        return True

    return False


_TRANSCRIPT_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")  # [crosstalk], [inaudible], [laughs], ...
_FILLER_WORDS = {"yeah", "um", "uh", "uhh", "umm", "mm", "hmm", "okay", "so", "well", "like"}


def _is_noisy_transcript_line(text: str) -> bool:
    """
    Real spoken-therapy transcripts (AnnoMI) carry disfluencies that make
    sense as speech but read as broken text when handed to a model as a
    written-chat style target: bracketed annotations ("[crosstalk]"), dashes
    marking speech cut off mid-word/thought ("-would like to", "And-"), and
    filler-word-dominated fragments ("Yeah, yeah, yeah."). None of these are
    a style worth imitating in a text chat — filtering them out at retrieval
    time (rather than re-ingesting) keeps the exemplar pool itself untouched.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _TRANSCRIPT_ANNOTATION_RE.search(stripped):
        return True
    if stripped.startswith("-") or stripped.endswith("-"):
        return True
    words = re.findall(r"[a-zA-Z']+", stripped.lower())
    if not words:
        return True
    filler_ratio = sum(1 for w in words if w in _FILLER_WORDS) / len(words)
    return filler_ratio > 0.35


def _lexical_overlap(query: str, doc: str) -> float:
    """Lightweight lexical signal to stabilize reranking for domain terms."""
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(doc)
    if not q_tokens or not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def _topic_bonus(mood: str, metadata: dict | None) -> float:
    if not metadata:
        return 0.0

    doc_topics = metadata.get("topics")
    if not doc_topics:
        return 0.0

    if isinstance(doc_topics, str):
        normalized = {t.strip().lower() for t in doc_topics.split(",") if t.strip()}
    elif isinstance(doc_topics, (list, tuple, set)):
        normalized = {str(t).strip().lower() for t in doc_topics if str(t).strip()}
    else:
        return 0.0

    mood_topics = _MOOD_TOPIC_HINTS.get((mood or "").lower(), set())
    if not mood_topics:
        return 0.0

    return 0.12 if normalized & mood_topics else 0.0


_shared_qdrant_client = None


def _get_shared_qdrant_client():
    """One QdrantClient for the whole process, shared across every collection.

    In local (embedded) mode Qdrant takes an EXCLUSIVE lock on the storage
    folder — not just across processes, but per client instance. Building a
    separate client per collection (clinical KB, style_exemplars,
    patient_memory) therefore made them fight each other for the same lock
    inside a single process.

    That was catastrophic for voice, and it was invisible because the failure
    only surfaced as a benign-looking "collection may not exist yet" warning.
    Measured live 2026-08-07: `_get_style_store()` — a SYNCHRONOUS call made
    directly on the event loop — blocked for 6-18s per turn waiting on that
    lock before giving up. That starved the asyncio loop, so /ws/voice
    couldn't forward mic audio, so Deepgram killed the STT socket with 1011
    ("did not receive audio data within the timeout window"). In the browser
    that appeared as "Speech recognition error", choppy playback, and huge
    latency. Because the failure was never cached, every single turn paid it
    again.

    Sharing one client removes the contention entirely and is also just
    correct: server mode already behaves this way (one HTTP client, no file
    lock), which is why this never appeared until local mode was enabled.
    """
    global _shared_qdrant_client
    if _shared_qdrant_client is None:
        from qdrant_client import QdrantClient
        if settings.QDRANT_MODE == "server":
            _shared_qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent
            _shared_qdrant_client = QdrantClient(
                path=str(project_root / settings.QDRANT_PATH.lstrip("./"))
            )
    return _shared_qdrant_client


class RAGService:
    def __init__(self, backend_override: str | None = None, redis_client=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        backend = (backend_override or settings.VECTOR_DB_BACKEND).lower()
        # clinical_kb — textbooks, guidelines, CBT/DBT manuals
        self.vector_store = self._build_vector_store(
            backend,
            collection=settings.QDRANT_COLLECTION_NAME,
            pinecone_index=settings.PINECONE_INDEX_NAME,
        )
        # style_exemplars — therapist transcript turns keyed on move + valence
        # Initialised lazily: the collection may not exist on first run.
        self._style_store = None
        # Style-exemplar results are a pure function of
        # (move, register, affect_valence, turn_position, k) — the
        # similarity_search query string is built from those enums alone and
        # contains ZERO user content, and the filter is metadata-only. That
        # makes every call cacheable, and the key space tiny (~9 moves x 2
        # registers x 3 valences). Measured live 2026-08-07: without this,
        # every voice turn spent 6.8-18.2s here (BGE-base CPU embedding +
        # a filtered local-Qdrant scan, run TWICE via the valence-relaxation
        # loop). That was both the dominant latency AND the cause of
        # Deepgram closing the STT socket with 1011 "did not receive audio
        # within the timeout window" — the GIL-holding CPU work starved the
        # event loop so the /ws/voice receive loop couldn't forward mic audio
        # for >10s, which surfaced in the browser as "Speech recognition
        # error" plus choppy playback.
        self._style_exemplar_cache: dict[tuple, list[dict[str, str]]] = {}
        self._style_backend = backend
        # patient_memory — per-user long-term semantic recall, written to
        # continuously as real sessions happen (not a one-time reference
        # corpus like clinical_kb/style_exemplars), so unlike those it's
        # created on first write rather than assumed to already exist.
        self._patient_memory_store = None
        # Cross-encoder loaded lazily — only used when ENABLE_RERANKER=true in env.
        self._reranker = None
        self._redis = redis_client
        self._local_cache: dict[str, tuple[float, str]] = {}

    async def _cache_get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                val = await self._redis.get(f"rag:{key}")
                return pickle.loads(val) if val else None
            except Exception:
                pass  # Redis unavailable — fall through to local cache
        entry = self._local_cache.get(key)
        if entry and (time.monotonic() - entry[0]) < _RAG_CACHE_TTL:
            return entry[1]
        return None

    async def _cache_set(self, key: str, value: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.setex(f"rag:{key}", _RAG_CACHE_TTL, pickle.dumps(value))
                return
            except Exception:
                pass  # Redis unavailable — fall through to local cache
        self._local_cache[key] = (time.monotonic(), value)

    def _build_vector_store(
        self,
        backend: str,
        collection: str | None = None,
        pinecone_index: str | None = None,
    ):
        """
        Factory: returns a LangChain vector store for the requested backend.
        Both PineconeVectorStore and QdrantVectorStore expose the identical
        max_marginal_relevance_search signature used in retrieve_clinical_context.
        """
        if backend == "pinecone":
            os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
            from langchain_pinecone import PineconeVectorStore
            idx = pinecone_index or settings.PINECONE_INDEX_NAME
            return PineconeVectorStore.from_existing_index(
                index_name=idx,
                embedding=self.embeddings,
            )

        if backend == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            col = collection or settings.QDRANT_COLLECTION_NAME
            client = _get_shared_qdrant_client()
            return QdrantVectorStore(
                client=client,
                collection_name=col,
                embedding=self.embeddings,
            )

        raise ValueError(
            f"Unknown VECTOR_DB_BACKEND: '{backend}'. Must be 'pinecone' or 'qdrant'."
        )

    def _get_style_store(self):
        """Lazy-initialise the style_exemplars vector store."""
        if self._style_store is None:
            try:
                self._style_store = self._build_vector_store(
                    self._style_backend,
                    collection=settings.QDRANT_STYLE_COLLECTION,
                    pinecone_index=settings.PINECONE_STYLE_INDEX_NAME,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Style exemplar store not available (collection may not exist yet): %s", e
                )
        return self._style_store

    async def retrieve_style_exemplars(
        self,
        move: str,
        register: str,
        affect_valence: str = "neutral",
        turn_position: str = "mid",
        k: int = 3,
    ) -> list[dict[str, str]]:
        """
        Retrieve therapist style exemplars keyed on move + valence + register.
        Retrieval is by metadata filter, NOT semantic similarity — content
        must not leak from one user's situation into another's response.

        `register` is a hard boundary and has no default — an exemplar in the
        wrong register (e.g. "en" vs "hinglish-casual") is worse than no
        exemplar, so this never falls back across registers, and there is
        deliberately no way to call this without picking one. `affect_valence`
        is a soft signal: today's exemplar sources have no real valence
        annotation, only a crude keyword heuristic over the preceding client
        turn, so most exemplars land in "neutral" regardless of actual mood.
        Requiring an exact valence match would make retrieval empty far too
        often, so valence is relaxed first if the strict filter finds nothing.

        Within a register, exemplars are also preferred by label confidence:
        expert-annotated sources (e.g. AnnoMI, no `label_confidence` field) are
        ranked ahead of machine-labeled ones (`label_confidence ==
        "machine_labeled"`, e.g. the self-authored Hinglish sessions). Today
        AnnoMI and the Hinglish set never share a register, so this has no
        observable effect yet — it exists so that adding a future source that
        DOES share a register with an existing lower/higher-confidence one
        (e.g. a expert-labeled Hindi corpus alongside the machine-labeled
        Hinglish sessions) doesn't silently blend confidence tiers with equal
        weight, ranked by nothing more than text similarity to the query.

        Returns list of {"patient": ..., "therapist": ...} pairs (empty list
        if store unavailable or nothing matches even after relaxing valence).
        Pairs, not bare therapist strings — dumping isolated therapist lines
        into the prompt as "context" reads as background knowledge to an
        RLHF-aligned model, not a style to imitate. Pairing each with the
        patient turn it actually responded to lets the prompt present them as
        genuine few-shot Input/Output demonstrations instead. Exemplars with
        no captured patient turn (e.g. a transcript's very first line) are
        dropped rather than emitted with a blank patient side.
        """
        cache_key = (move, register, affect_valence, turn_position, k)
        cached = self._style_exemplar_cache.get(cache_key)
        if cached is not None:
            return cached

        store = self._get_style_store()
        if store is None:
            return []

        def _filter(include_valence: bool):
            # LangChain does NOT normalise `filter=` across vector store
            # backends — each expects its own native shape. Qdrant needs a
            # real Filter/FieldCondition object (a plain dict fails pydantic
            # validation with "Extra inputs are not permitted"), and its
            # payload nests our metadata under a "metadata" key.
            if self._style_backend == "qdrant":
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                conditions = [
                    FieldCondition(key="metadata.move", match=MatchValue(value=move)),
                    FieldCondition(key="metadata.register", match=MatchValue(value=register)),
                ]
                if include_valence:
                    conditions.append(
                        FieldCondition(key="metadata.affect_valence", match=MatchValue(value=affect_valence))
                    )
                return Filter(must=conditions)

            # Pinecone's native filter shape.
            return {
                "move": {"$eq": move},
                "register": {"$eq": register},
                **({"affect_valence": {"$eq": affect_valence}} if include_valence else {}),
            }

        try:
            for include_valence in (True, False):
                docs = await asyncio.to_thread(
                    store.similarity_search,
                    query=f"{move} {affect_valence} {turn_position}",
                    k=k * 8,  # over-fetch — trimmed further below by pairing + noise filtering
                    filter=_filter(include_valence),
                )
                # Expert-labeled first, machine-labeled last — see docstring.
                docs.sort(key=lambda d: (d.metadata or {}).get("label_confidence") == "machine_labeled")
                results = []
                for doc in docs:
                    therapist_response = (doc.page_content or "").strip()
                    patient_input = ((doc.metadata or {}).get("client_context_snippet") or "").strip()
                    if not therapist_response or not patient_input:
                        continue
                    if _is_noisy_transcript_line(therapist_response) or _is_noisy_transcript_line(patient_input):
                        continue
                    results.append({"patient": patient_input, "therapist": therapist_response})
                    if len(results) >= k:
                        break
                if results:
                    self._style_exemplar_cache[cache_key] = results
                    return results
            # Cache the empty result too — a miss costs exactly as much to
            # recompute as a hit (two full filtered searches), and "this
            # move/register combination has no exemplars" is just as stable
            # a fact as a positive result.
            self._style_exemplar_cache[cache_key] = []
            return []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Style exemplar retrieval failed: %s", e)
            return []

    def _get_patient_memory_store(self):
        """Lazy-initialise the patient_memory store for READS. Returns None if
        it doesn't exist yet (no session has ever been consolidated) — callers
        must treat that as 'no memories yet', not an error."""
        if self._patient_memory_store is None:
            try:
                self._patient_memory_store = self._build_vector_store(
                    self._style_backend,
                    collection=settings.QDRANT_PATIENT_MEMORY_COLLECTION,
                    pinecone_index=settings.QDRANT_PATIENT_MEMORY_COLLECTION,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).info(
                    "Patient memory store not available yet (no sessions consolidated): %s", e
                )
        return self._patient_memory_store

    async def store_patient_memory(
        self, user_id: str, session_id: str, content: str, chunk_type: str = "dialogue_highlight",
    ) -> None:
        """
        Append one long-term memory chunk for this user. Unlike clinical_kb/
        style_exemplars (one-time reference corpora, ingested by a script that
        assumes the collection doesn't exist), this collection accumulates
        continuously across real sessions, so it's created on first write
        here rather than by a separate ingestion step — and only CREATED,
        never recreated, so an existing user's memories are never wiped by a
        later write.
        """
        text = (content or "").strip()
        if not text:
            return
        try:
            from langchain_core.documents import Document
            doc = Document(
                page_content=text,
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "chunk_type": chunk_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            def _write():
                if self._style_backend == "qdrant":
                    from qdrant_client.models import Distance, VectorParams
                    from langchain_qdrant import QdrantVectorStore

                    # Shared client — a second one would deadlock against the
                    # embedded-mode storage lock (see _get_shared_qdrant_client).
                    client = _get_shared_qdrant_client()

                    collection = settings.QDRANT_PATIENT_MEMORY_COLLECTION
                    if not client.collection_exists(collection):
                        vector_size = len(self.embeddings.embed_query("dimension probe"))
                        client.create_collection(
                            collection_name=collection,
                            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                        )
                    store = QdrantVectorStore(client=client, collection_name=collection, embedding=self.embeddings)
                else:
                    store = self._build_vector_store(
                        self._style_backend, collection=settings.QDRANT_PATIENT_MEMORY_COLLECTION,
                    )
                store.add_documents([doc])
                return store

            self._patient_memory_store = await asyncio.to_thread(_write)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("store_patient_memory failed for user=%s: %s", user_id, e)

    async def retrieve_patient_memory(self, user_id: str, query: str, k: int = 3) -> list[str]:
        """
        Semantic search over this one user's own past dialogue highlights.
        `user_id` is a hard filter, never relaxed — leaking one user's
        memories into another's context is a privacy violation, not an
        accuracy tradeoff (same principle as `register` in
        retrieve_style_exemplars, applied to identity instead of language).
        """
        store = self._get_patient_memory_store()
        if store is None:
            return []
        try:
            if self._style_backend == "qdrant":
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                filter_ = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
            else:
                filter_ = {"user_id": {"$eq": user_id}}
            docs = await asyncio.to_thread(
                store.similarity_search, query=query, k=k, filter=filter_,
            )
            return [d.page_content.strip() for d in docs if d.page_content and d.page_content.strip()]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("retrieve_patient_memory failed for user=%s: %s", user_id, e)
            return []

    def _expand_with_idioms(self, query: str) -> str:
        """
        Scan query text for South Asian distress idioms and return
        additional clinical search terms to append to the expanded query.
        """
        query_lower = query.lower()
        extra: set[str] = set()
        for idiom, terms in _IDIOM_EXPANSIONS.items():
            if idiom in query_lower:
                extra |= terms
        return " ".join(extra)

    async def retrieve_clinical_context(self, query: str, mood: str) -> str:
        """
        Precision-first retrieval pipeline:
          1. Two MMR searches in parallel (raw and mood-expanded query)
          2. Candidate dedup + noise filtering
          3. Cross-encoder + lexical hybrid reranking
          4. Adaptive top-k context assembly
        Results are cached for _RAG_CACHE_TTL seconds to avoid redundant
        cross-encoder inference for repeated or near-identical queries.
        """
        cache_key = hashlib.md5(f"{query}|{(mood or '').lower()}".encode()).hexdigest()
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached

        expansion = _MOOD_EXPANSION.get((mood or "").lower(), _DEFAULT_EXPANSION)
        # Supplement with idiom-specific terms when South Asian phrases are present.
        idiom_terms = self._expand_with_idioms(query)
        expanded_query = f"{query} {expansion} {idiom_terms}".strip()

        raw_docs, expanded_docs = await asyncio.gather(
            asyncio.to_thread(
                self.vector_store.max_marginal_relevance_search,
                query, k=_MMR_K, fetch_k=_MMR_FETCH_K,
            ),
            asyncio.to_thread(
                self.vector_store.max_marginal_relevance_search,
                expanded_query, k=_MMR_K, fetch_k=_MMR_FETCH_K,
            ),
        )

        # Merge while preserving order and removing near-duplicates by normalized content.
        docs = []
        seen = set()
        for doc in [*(raw_docs or []), *(expanded_docs or [])]:
            key = " ".join((doc.page_content or "").split()).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            docs.append(doc)

        docs = [d for d in docs if not _is_low_signal_chunk(d.page_content)]

        if not docs:
            return ""

        # Cross-encoder reranking is opt-in (set ENABLE_RERANKER=true in .env).
        # On CPU it costs 1-3s per request; MMR cosine + lexical + topic scoring
        # gives sufficient precision for sub-2s latency with this KB size.
        if os.getenv("ENABLE_RERANKER", "false").lower() == "true":
            if self._reranker is None:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(RERANK_MODEL)
            rerank_docs = docs[:_MAX_RERANK_DOCS]
            pairs = [[query, d.page_content] for d in rerank_docs]
            cross_scores = list(await asyncio.to_thread(self._reranker.predict, pairs))
            cross_scores += [0.0] * (len(docs) - len(rerank_docs))
        else:
            cross_scores = [0.0] * len(docs)

        scored_docs = []
        for cross_score, doc in zip(cross_scores, docs):
            lexical = _lexical_overlap(query, doc.page_content)
            topical = _topic_bonus(mood, doc.metadata)
            # Cross-encoder is primary when available; lexical signal reduces off-topic passages.
            hybrid_score = float(cross_score) + (0.25 * lexical) + topical
            scored_docs.append((hybrid_score, float(cross_score), lexical, topical, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        best = scored_docs[0][0]
        # Keep candidates near the top score; cap to avoid long noisy context blocks.
        min_keep = max(best - 0.30, 0.15)
        selected = [item for item in scored_docs if item[0] >= min_keep][: _MAX_FINAL_DOCS]

        if not selected:
            selected = scored_docs[:2]

        # Add source/page anchors when metadata exists to help grounded generation.
        rendered = []
        for _, _, _, _, doc in selected:
            source = doc.metadata.get("source") if doc.metadata else None
            page = doc.metadata.get("page") if doc.metadata else None
            header = ""
            if source is not None:
                header = f"[source: {Path(str(source)).name}"
                if page is not None:
                    header += f", page: {page}]\n"
                else:
                    header += "]\n"
            rendered.append(f"{header}{doc.page_content}".strip())

        result = "\n\n".join(rendered)
        await self._cache_set(cache_key, result)
        return result
