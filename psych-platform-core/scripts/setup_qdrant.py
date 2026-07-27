"""
Creates (or recreates) the local Qdrant collection for psych-brain RAG.
Run once before scripts/ingest.py when VECTOR_DB_BACKEND=qdrant.

Usage:
    python scripts/setup_qdrant.py              # create if not exists
    python scripts/setup_qdrant.py --recreate   # drop and recreate
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from app.core.config import settings

DIMENSION = 768  # bge-base-en-v1.5 output dimension


def setup_collection(recreate: bool = False):
    project_root = Path(__file__).resolve().parent.parent
    qdrant_path = str(project_root / settings.QDRANT_PATH.lstrip("./"))
    collection_name = settings.QDRANT_COLLECTION_NAME

    print(f"Qdrant local storage : {qdrant_path}")
    print(f"Collection name      : {collection_name}")

    client = QdrantClient(path=qdrant_path)
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            print(f"Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print("Deleted.")
        else:
            info = client.get_collection(collection_name)
            print(
                f"Collection '{collection_name}' already exists "
                f"({info.points_count} vectors). Nothing to do."
            )
            print("Use --recreate to drop and recreate.")
            return

    print(f"Creating collection '{collection_name}' (dim={DIMENSION}, metric=cosine)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' is ready.")
    print("\nNext step: python scripts/ingest.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up local Qdrant collection")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection (WARNING: deletes all vectors)",
    )
    args = parser.parse_args()
    setup_collection(recreate=args.recreate)
