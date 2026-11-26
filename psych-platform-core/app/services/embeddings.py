"""
Embedding backends for RAGService.

Two interchangeable backends produce the SAME vectors (BAAI/bge-base-en-v1.5,
L2-normalised, 768-d), so an index built with one can be queried with the other:

  local   — in-process via sentence-transformers / PyTorch. Needs ~1 GB RAM.
  hf_api  — Hugging Face hosted inference (`hf-inference` provider). No torch in
            the process, so the app fits a 512 MB host such as Render's free tier.

Selected by settings.EMBEDDING_BACKEND. Everything here is synchronous on
purpose: LangChain vector stores call embed_query() from a worker thread (the
RAG service wraps every search in asyncio.to_thread), exactly as they do for
the local backend.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, List

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

# Hosted inference occasionally 503s while a serverless model warms up. One quick
# retry absorbs that; more would hold a worker thread (and the user) for too long.
_MAX_RETRIES = 1
_RETRY_BACKOFF_S = 0.75
_REQUEST_TIMEOUT_S = 10.0
_BATCH_SIZE = 32
_QUERY_CACHE_SIZE = 128


class HFInferenceEmbeddings(Embeddings):
    """LangChain Embeddings backed by Hugging Face hosted inference."""

    def __init__(self, model_name: str, token: str, client: Any = None):
        if client is None:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                provider="hf-inference", api_key=token, timeout=_REQUEST_TIMEOUT_S
            )
        self.model_name = model_name
        self._client = client
        # Repeated/near-identical queries are common (retries, the mood-expanded
        # and raw variants of one message); a tiny LRU saves a network round trip.
        self._query_cache: "OrderedDict[str, List[float]]" = OrderedDict()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _to_matrix(raw: Any, n_inputs: int) -> np.ndarray:
        """Coerce the API response to an (n_inputs, dim) L2-normalised matrix."""
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3:
            # Token-level output (batch, tokens, dim): BGE pools with the CLS token.
            arr = arr[:, 0, :]
        elif arr.ndim == 2 and n_inputs == 1 and arr.shape[0] != 1:
            # A single input returned token-level (tokens, dim): take CLS.
            arr = arr[:1]
        if arr.ndim != 2 or arr.shape[0] != n_inputs:
            raise ValueError(
                f"Unexpected embedding response shape {arr.shape} for {n_inputs} input(s)"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    def _embed(self, inputs: "str | List[str]") -> np.ndarray:
        n = 1 if isinstance(inputs, str) else len(inputs)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                raw = self._client.feature_extraction(inputs, model=self.model_name)
                return self._to_matrix(raw, n)
            except Exception as exc:  # network, 4xx/5xx, quota, bad shape
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
        logger.error("HF embedding request failed (%s): %s", type(last_exc).__name__, last_exc)
        raise RuntimeError("Hugging Face embedding request failed") from last_exc

    # -- Embeddings interface ---------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            return list(cached)
        vec = self._embed(text)[0].tolist()
        self._query_cache[text] = vec
        if len(self._query_cache) > _QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return list(vec)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            out.extend(self._embed(texts[i : i + _BATCH_SIZE]).tolist())
        return out


def build_embeddings(model_name: str) -> Embeddings:
    """Return the embedding backend selected by settings.EMBEDDING_BACKEND."""
    if settings.EMBEDDING_BACKEND == "hf_api":
        logger.info("Embeddings backend: Hugging Face hosted inference (%s)", model_name)
        return HFInferenceEmbeddings(model_name, settings.HF_API_TOKEN)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
