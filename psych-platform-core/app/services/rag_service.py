import os
import asyncio
import hashlib
import pickle
import re
import time
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


class RAGService:
    def __init__(self, backend_override: str | None = None, redis_client=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        backend = (backend_override or settings.VECTOR_DB_BACKEND).lower()
        self.vector_store = self._build_vector_store(backend)
        # Cross-encoder loaded lazily — only used when ENABLE_RERANKER=true in env.
        # Default off: on CPU it adds 1-3s per request; MMR + lexical scoring is
        # sufficient for sub-2s latency targets.
        self._reranker = None
        # Redis client for shared cross-process cache.  None → fall back to the
        # in-process dict (safe for single-worker dev).
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

    def _build_vector_store(self, backend: str):
        """
        Factory: returns a LangChain vector store for the requested backend.
        Both PineconeVectorStore and QdrantVectorStore expose the identical
        max_marginal_relevance_search signature used in retrieve_clinical_context.
        """
        if backend == "pinecone":
            os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
            from langchain_pinecone import PineconeVectorStore
            return PineconeVectorStore.from_existing_index(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
            )

        if backend == "qdrant":
            from qdrant_client import QdrantClient
            from langchain_qdrant import QdrantVectorStore
            if settings.QDRANT_MODE == "server":
                # Docker / multi-worker: connects over HTTP — no file lock.
                # Start with: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
                client = QdrantClient(url=settings.QDRANT_URL)
            else:
                # Local file-based (single-process dev only).
                project_root = Path(__file__).resolve().parent.parent.parent
                qdrant_path = str(project_root / settings.QDRANT_PATH.lstrip("./"))
                client = QdrantClient(path=qdrant_path)
            return QdrantVectorStore(
                client=client,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embedding=self.embeddings,
            )

        raise ValueError(
            f"Unknown VECTOR_DB_BACKEND: '{backend}'. Must be 'pinecone' or 'qdrant'."
        )

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
        expanded_query = f"{query} {expansion}".strip()

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
