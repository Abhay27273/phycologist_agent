"""Unit tests for the pieces that let the app run on a 512 MB host:
hosted embeddings (app/services/embeddings.py), the DB URL sanitizer, and the
EMBEDDING_BACKEND config guard. No network, no model downloads."""
import numpy as np
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.infrastructure.database import get_sanitized_url
from app.services import embeddings as emb
from app.services.embeddings import HFInferenceEmbeddings


class FakeClient:
    """Stands in for huggingface_hub.InferenceClient."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def feature_extraction(self, inputs, model=None):
        self.calls.append((inputs, model))
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(emb.time, "sleep", lambda _s: None)


def _norm(v):
    return float(np.linalg.norm(v))


# ---- HFInferenceEmbeddings ---------------------------------------------------

def test_query_vector_is_l2_normalised():
    client = FakeClient([np.array([3.0, 4.0] + [0.0] * 766, dtype=np.float32)])
    e = HFInferenceEmbeddings("m", "tok", client=client)
    vec = e.embed_query("hello")
    assert len(vec) == 768
    assert _norm(vec) == pytest.approx(1.0, abs=1e-5)
    assert vec[0] == pytest.approx(0.6, abs=1e-5)
    assert client.calls == [("hello", "m")]


def test_token_level_output_is_pooled_with_cls_for_single_and_batch():
    cls = np.zeros(8, dtype=np.float32)
    cls[0] = 2.0
    other = np.ones(8, dtype=np.float32)
    token_level = np.stack([cls, other, other])  # (tokens, dim) for ONE input
    e = HFInferenceEmbeddings("m", "tok", client=FakeClient([token_level]))
    vec = e.embed_query("x")
    assert vec[0] == pytest.approx(1.0, abs=1e-5)  # CLS row only, normalised

    batch = np.stack([token_level, token_level])  # (2, tokens, dim)
    e2 = HFInferenceEmbeddings("m", "tok", client=FakeClient([batch]))
    out = e2.embed_documents(["a", "b"])
    assert len(out) == 2 and out[0][0] == pytest.approx(1.0, abs=1e-5)


def test_retries_once_then_succeeds():
    good = np.ones(4, dtype=np.float32)
    client = FakeClient([RuntimeError("503 warming up"), good])
    e = HFInferenceEmbeddings("m", "tok", client=client)
    assert len(e.embed_query("q")) == 4
    assert len(client.calls) == 2


def test_raises_after_retries_exhausted():
    client = FakeClient([RuntimeError("down"), RuntimeError("still down")])
    e = HFInferenceEmbeddings("m", "tok", client=client)
    with pytest.raises(RuntimeError, match="embedding request failed"):
        e.embed_query("q")
    assert len(client.calls) == 2  # 1 attempt + 1 retry, no more


def test_wrong_row_count_is_an_error_not_silent_garbage():
    # 3 vectors back for 2 inputs must not be zipped into the wrong documents.
    client = FakeClient([np.ones((3, 4), dtype=np.float32)] * 2)
    e = HFInferenceEmbeddings("m", "tok", client=client)
    with pytest.raises(RuntimeError):
        e.embed_documents(["a", "b"])


def test_query_cache_avoids_second_call_and_returns_copies():
    client = FakeClient([np.ones(4, dtype=np.float32)])
    e = HFInferenceEmbeddings("m", "tok", client=client)
    first = e.embed_query("same")
    first[0] = 999.0  # caller mutating its copy must not poison the cache
    second = e.embed_query("same")
    assert len(client.calls) == 1
    assert second[0] != 999.0


def test_embed_documents_batches_requests():
    n = emb._BATCH_SIZE + 5
    client = FakeClient([
        np.ones((emb._BATCH_SIZE, 4), dtype=np.float32),
        np.ones((5, 4), dtype=np.float32),
    ])
    e = HFInferenceEmbeddings("m", "tok", client=client)
    out = e.embed_documents([f"t{i}" for i in range(n)])
    assert len(out) == n
    assert [len(c[0]) for c in client.calls] == [emb._BATCH_SIZE, 5]


# ---- RAGService._qdrant_store (hosted-embedding boot safety) ------------------

class _FakeQdrant:
    def __init__(self, exists):
        self._exists = exists

    def collection_exists(self, name):
        return self._exists


class _ExplodingEmbeddings:
    """Any embedding call during store construction is a boot-time network call."""

    def embed_documents(self, texts):
        raise AssertionError("embedding call during store construction")

    def embed_query(self, text):
        raise AssertionError("embedding call during store construction")


def _bare_rag(monkeypatch, backend):
    from app.services import rag_service as rs
    monkeypatch.setattr(rs.settings, "EMBEDDING_BACKEND", backend)
    svc = object.__new__(rs.RAGService)  # skip __init__: no model / DB needed
    svc.embeddings = _ExplodingEmbeddings()
    return svc


def test_hosted_mode_builds_store_without_any_embedding_call(monkeypatch):
    svc = _bare_rag(monkeypatch, "hf_api")
    store = svc._qdrant_store(_FakeQdrant(exists=True), "psych-brain")
    assert store.collection_name == "psych-brain"


def test_hosted_mode_still_fails_for_a_missing_collection(monkeypatch):
    # Style-exemplar / patient-memory code depends on this raising to mean
    # "unavailable" (and to skip the doomed warm-up sweep).
    svc = _bare_rag(monkeypatch, "hf_api")
    with pytest.raises(ValueError, match="does not exist"):
        svc._qdrant_store(_FakeQdrant(exists=False), "psych-style")


# ---- DB URL sanitizer ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("postgresql://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
    ("postgres://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
    ("postgresql+asyncpg://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
    ("postgresql://u:p@localhost/db", "postgresql+asyncpg://u:p@127.0.0.1/db"),
    ("sqlite+aiosqlite:///./psych_db.sqlite", "sqlite+aiosqlite:///./psych_db.sqlite"),
])
def test_sanitized_url_driver_and_host(raw, expected):
    assert get_sanitized_url(raw) == expected


def test_sslmode_becomes_ssl_and_channel_binding_is_dropped():
    out = get_sanitized_url("postgresql://u:p@h/db?sslmode=require&channel_binding=require")
    assert out == "postgresql+asyncpg://u:p@h/db?ssl=require"


def test_other_query_params_survive():
    out = get_sanitized_url("postgresql://u:p@h/db?sslmode=require&application_name=haven")
    assert "ssl=require" in out and "application_name=haven" in out
    assert "sslmode" not in out


def test_sanitizer_is_idempotent():
    once = get_sanitized_url("postgresql://u:p@h/db?sslmode=require")
    assert get_sanitized_url(once) == once


# ---- config guard -------------------------------------------------------------

_BASE = dict(DATABASE_URL="postgresql://u:p@h/db", GOOGLE_API_KEY="k", VECTOR_DB_BACKEND="qdrant")


def test_hf_api_backend_requires_a_token():
    with pytest.raises(ValidationError, match="HF_API_TOKEN"):
        Settings(_env_file=None, EMBEDDING_BACKEND="hf_api", HF_API_TOKEN="", **_BASE)


def test_hf_api_backend_ok_with_token_and_local_needs_none():
    assert Settings(_env_file=None, EMBEDDING_BACKEND="hf_api", HF_API_TOKEN="hf_x", **_BASE)
    assert Settings(_env_file=None, EMBEDDING_BACKEND="local", **_BASE).QDRANT_API_KEY == ""
