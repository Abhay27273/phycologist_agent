"""
Ingest mental-health counseling datasets from HuggingFace into the Qdrant
knowledge base.  Run after ingest.py (PDF clinical texts) to supplement
the vector store with real counseling Q&A pairs.

Usage:
    python scripts/ingest_datasets.py

Requires:
    pip install datasets

Datasets indexed:
  - Amod/mental_health_counseling_conversations  (~2 k Q&A pairs, high quality)
  - nbertagnolli/counsel-chat                    (~930 counseling transcripts)

Both datasets are filtered for minimum answer length so only substantive
therapeutic content enters the knowledge base.
"""
import os
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
MIN_ANSWER_CHARS = 120
BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Dataset definitions
# Each entry must provide 'name', 'split', and field names for question/answer.
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "Amod/mental_health_counseling_conversations",
        "split": "train",
        "question_field": "Context",
        "answer_field": "Response",
        "source_type": "counseling_qa",
    },
    {
        "name": "nbertagnolli/counsel-chat",
        "split": "train",
        "question_field": "questionBody",
        "answer_field": "answerText",
        "source_type": "counseling_transcript",
    },
]


def _clean(text: str) -> str:
    if not text:
        return ""
    # Collapse whitespace, strip Reddit/HTML noise
    text = re.sub(r"\[.*?\]|\(https?://\S+\)", "", text)
    return " ".join(text.split())


def _load_dataset_docs(cfg: dict) -> list[Document]:
    """Download one HuggingFace dataset and convert rows to LangChain Documents."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"  Downloading {cfg['name']} ({cfg['split']})...")
    try:
        ds = load_dataset(cfg["name"], split=cfg["split"], trust_remote_code=True)
    except Exception as e:
        print(f"  SKIP {cfg['name']}: {e}")
        return []

    docs = []
    skipped = 0
    for row in ds:
        question = _clean(row.get(cfg["question_field"], "") or "")
        answer = _clean(row.get(cfg["answer_field"], "") or "")

        if len(answer) < MIN_ANSWER_CHARS:
            skipped += 1
            continue

        content = f"Question: {question}\n\nAnswer: {answer}" if question else answer
        docs.append(Document(
            page_content=content,
            metadata={
                "source": cfg["name"],
                "source_type": cfg["source_type"],
                "quality": "high",
            },
        ))

    print(f"  Loaded {len(docs)} docs (skipped {skipped} too-short answers)")
    return docs


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isalpha() for ch in text) / len(text)


def ingest_datasets():
    print("=== HuggingFace Dataset Ingestion ===\n")

    # --- 1. Load all datasets ---
    all_docs: list[Document] = []
    for cfg in DATASETS:
        print(f"--- Dataset: {cfg['name']} ---")
        docs = _load_dataset_docs(cfg)
        all_docs.extend(docs)
    print(f"\nTotal raw documents: {len(all_docs)}")

    if not all_docs:
        print("No documents loaded. Exiting.")
        return

    # --- 2. Split ---
    print("\n--- Splitting text ---")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_splits = splitter.split_documents(all_docs)

    # Filter low-signal chunks
    splits = [
        c for c in raw_splits
        if len(c.page_content.strip()) >= 140 and _alpha_ratio(c.page_content) >= 0.6
    ]
    print(f"Chunks: {len(raw_splits)} raw → {len(splits)} kept")

    if not splits:
        print("No usable chunks after filtering. Exiting.")
        return

    # --- 3. Embed ---
    print("\n--- Loading embedding model ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- 4. Index into Qdrant ---
    backend = settings.VECTOR_DB_BACKEND.lower()
    print(f"\n--- Indexing {len(splits)} chunks into {backend.upper()} ---")

    if backend == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore

        if getattr(settings, "QDRANT_MODE", "local") == "server":
            client = QdrantClient(url=settings.QDRANT_URL)
            print(f"  Qdrant server: {settings.QDRANT_URL}")
        else:
            project_root = Path(__file__).resolve().parent.parent
            qdrant_path = str(project_root / settings.QDRANT_PATH.lstrip("./"))
            client = QdrantClient(path=qdrant_path)
            print(f"  Qdrant local: {qdrant_path}")

        # Ensure collection exists
        existing = {c.name for c in client.get_collections().collections}
        if settings.QDRANT_COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"  Created collection '{settings.QDRANT_COLLECTION_NAME}'")
        else:
            print(f"  Appending to existing collection '{settings.QDRANT_COLLECTION_NAME}'")

        vs = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
        total = len(splits)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, total, BATCH_SIZE):
            batch = splits[i: i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")
            try:
                vs.add_documents(batch)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")

        print(f"\nSUCCESS: {len(splits)} counseling chunks added to Qdrant.")

    elif backend == "pinecone":
        os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        from langchain_pinecone import PineconeVectorStore
        total = len(splits)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, total, BATCH_SIZE):
            batch = splits[i: i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            print(f"  Batch {batch_num}/{total_batches}...", end=" ")
            try:
                PineconeVectorStore.from_documents(batch, embeddings, index_name=settings.PINECONE_INDEX_NAME)
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
        print(f"\nSUCCESS: {len(splits)} counseling chunks added to Pinecone.")

    else:
        print(f"ERROR: Unknown VECTOR_DB_BACKEND '{backend}'.")


if __name__ == "__main__":
    ingest_datasets()
