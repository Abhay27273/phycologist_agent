import os
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

_TOPIC_KEYWORDS = {
    "anxiety": ["anxiety", "panic", "worry", "social anxiety", "phobia"],
    "depression": ["depression", "hopeless", "anhedonia", "low mood", "worthless"],
    "relationship": ["partner", "relationship", "argument", "conflict", "attachment"],
    "stress": ["stress", "burnout", "overwhelmed", "pressure", "coping"],
    "crisis": ["suicide", "self-harm", "ideation", "safety plan", "crisis"],
}


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / len(text)


def _looks_like_form(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "add columns",
        "not difficult at all",
        "healthcare professional",
        "total:",
        "0 1 2 3",
        "score",
        "questionnaire",
        "check all that apply",
    ]
    if any(marker in lowered for marker in markers):
        return True

    # A lot of checklist-like lines tends to indicate psychometric forms, not therapeutic guidance.
    checklist_lines = 0
    for line in (text or "").splitlines():
        compact = line.strip().lower()
        if re.match(r"^(\d+\.|[\-\*])\s+", compact):
            checklist_lines += 1
    return checklist_lines >= 8


def _infer_source_type(source: str) -> str:
    lowered = source.lower()
    if "guidelines" in lowered:
        return "guideline"
    if "instrument" in lowered or "phq" in lowered or "gad" in lowered or "cssrs" in lowered:
        return "instrument"
    if "taxonomy" in lowered:
        return "taxonomy"
    return "textbook"


def _infer_topics(text: str, source: str) -> list[str]:
    lowered = f"{source} {text}".lower()
    topics = []
    for topic, words in _TOPIC_KEYWORDS.items():
        if any(w in lowered for w in words):
            topics.append(topic)
    return topics


def _should_keep_page(doc) -> tuple[bool, str]:
    text = _normalize(doc.page_content)
    if len(text) < 180:
        return False, "too_short"
    if _alpha_ratio(text) < 0.6:
        return False, "low_alpha_ratio"
    if _looks_like_form(text):
        return False, "form_like"
    return True, "ok"


def _enrich_page_metadata(doc) -> None:
    source = str(doc.metadata.get("source", ""))
    normalized = _normalize(doc.page_content)
    doc.metadata["source_type"] = _infer_source_type(source)
    doc.metadata["quality"] = "high"
    doc.metadata["topics"] = _infer_topics(normalized, source)
    doc.metadata["char_count"] = len(normalized)


def _enrich_chunk_metadata(chunk) -> None:
    source = str(chunk.metadata.get("source", ""))
    normalized = _normalize(chunk.page_content)
    chunk.metadata["source_type"] = chunk.metadata.get("source_type") or _infer_source_type(source)
    chunk.metadata["topics"] = chunk.metadata.get("topics") or _infer_topics(normalized, source)
    chunk.metadata["quality"] = "high"
    chunk.metadata["chunk_char_count"] = len(normalized)


def _prepare_pages(docs):
    kept = []
    dropped = {"too_short": 0, "low_alpha_ratio": 0, "form_like": 0}

    for doc in docs:
        keep, reason = _should_keep_page(doc)
        if not keep:
            dropped[reason] += 1
            continue
        _enrich_page_metadata(doc)
        kept.append(doc)

    return kept, dropped


def _prepare_chunks(chunks):
    ready = []
    dropped = 0
    for chunk in chunks:
        normalized = _normalize(chunk.page_content)
        if len(normalized) < 140 or _alpha_ratio(normalized) < 0.6 or _looks_like_form(normalized):
            dropped += 1
            continue
        _enrich_chunk_metadata(chunk)
        ready.append(chunk)
    return ready, dropped


def _build_splitter(embeddings):
    splitter_mode = os.getenv("INGEST_SPLITTER", "semantic").strip().lower()

    if splitter_mode in {"recursive", "char", "character"}:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("  Using RecursiveCharacterTextSplitter (INGEST_SPLITTER=recursive)")
        return RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        print("  Using SemanticChunker (concept-aware splits)")
        return SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("  Falling back to RecursiveCharacterTextSplitter (pip install langchain-experimental for better chunking)")
        return RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


def ingest_psychology_data():
    data_path = Path("data")
    if not data_path.exists():
        os.makedirs(data_path)
        print("Created 'data/' folder. Add PDF books there and run again.")
        return

    print("--- 1. Loading PDFs ---")
    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        print("No PDFs found in 'data/'. Skipping.")
        return
    print(f"Loaded {len(docs)} pages.")

    docs, dropped_pages = _prepare_pages(docs)
    print(
        "Filtered pages | kept={} dropped={} (too_short={}, low_alpha_ratio={}, form_like={})".format(
            len(docs),
            sum(dropped_pages.values()),
            dropped_pages["too_short"],
            dropped_pages["low_alpha_ratio"],
            dropped_pages["form_like"],
        )
    )

    if not docs:
        print("No high-quality pages remained after filtering. Skipping.")
        return

    print("--- 2. Loading embedding model ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("--- 3. Splitting text ---")
    splitter = _build_splitter(embeddings)
    raw_splits = splitter.split_documents(docs)
    splits, dropped_chunks = _prepare_chunks(raw_splits)
    total_chunks = len(splits)
    print(f"Created {len(raw_splits)} chunks; kept {total_chunks}, dropped {dropped_chunks} low-signal chunks.")

    if not splits:
        print("No high-quality chunks remained after filtering. Skipping.")
        return

    backend = settings.VECTOR_DB_BACKEND.lower()
    print(f"--- 4. Embedding & indexing into {backend.upper()} ---")
    batch_size = 100

    if backend == "pinecone":
        os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        from langchain_pinecone import PineconeVectorStore
        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            try:
                PineconeVectorStore.from_documents(
                    batch, embeddings, index_name=settings.PINECONE_INDEX_NAME
                )
                print(f"  OK batch {batch_num}")
            except Exception as e:
                print(f"  ERROR batch {batch_num}: {e}")
        print("SUCCESS: Psychology knowledge indexed into Pinecone.")

    elif backend == "qdrant":
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        if getattr(settings, "QDRANT_MODE", "local") == "server":
            client = QdrantClient(url=settings.QDRANT_URL)
            print(f"  Using Qdrant server at {settings.QDRANT_URL}")
        else:
            project_root = Path(__file__).resolve().parent.parent
            qdrant_path = str(project_root / settings.QDRANT_PATH.lstrip("./"))
            client = QdrantClient(path=qdrant_path)
        # Ensure the collection exists before add_documents (which doesn't auto-create)
        from qdrant_client.models import Distance, VectorParams
        existing = {c.name for c in client.get_collections().collections}
        if settings.QDRANT_COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"  Created collection '{settings.QDRANT_COLLECTION_NAME}'")
        else:
            print(f"  Collection '{settings.QDRANT_COLLECTION_NAME}' already exists — appending")
        vs = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
        for i in range(0, total_chunks, batch_size):
            batch = splits[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            try:
                vs.add_documents(batch)
                print(f"  OK batch {batch_num}")
            except Exception as e:
                print(f"  ERROR batch {batch_num}: {e}")
        print("SUCCESS: Psychology knowledge indexed into Qdrant.")

    else:
        print(f"ERROR: Unknown VECTOR_DB_BACKEND '{backend}'. Set to 'pinecone' or 'qdrant' in .env.")


if __name__ == "__main__":
    ingest_psychology_data()
