"""
Deletes and recreates the Pinecone index with the correct dimension for
bge-base-en-v1.5 (768), then re-runs ingestion.
"""
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

DIMENSION = 768        # bge-base-en-v1.5
METRIC    = "cosine"
CLOUD     = "aws"
REGION    = "us-east-1"


def _should_force_recreate() -> bool:
    return os.getenv("FORCE_RECREATE_INDEX", "0").strip().lower() in {"1", "true", "yes", "y"}


def recreate_index():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME
    force_recreate = _should_force_recreate()

    existing = [i.name for i in pc.list_indexes()]

    if index_name in existing:
        current_dim = pc.describe_index(index_name).dimension
        if current_dim == DIMENSION and not force_recreate:
            print(f"Index '{index_name}' already has dimension {DIMENSION}. Nothing to do.")
            return
        print(f"Deleting index '{index_name}' (dim={current_dim}) ...")
        pc.delete_index(index_name)
        # Wait for deletion to propagate
        for _ in range(30):
            if index_name not in [i.name for i in pc.list_indexes()]:
                break
            time.sleep(2)
        print("Deleted.")

    print(f"Creating index '{index_name}' (dim={DIMENSION}, metric={METRIC}) ...")
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )

    # Wait until ready
    for _ in range(30):
        status = pc.describe_index(index_name).status
        if status.get("ready"):
            break
        print("  Waiting for index to be ready...")
        time.sleep(3)

    print(f"Index '{index_name}' is ready with dimension {DIMENSION}.")


if __name__ == "__main__":
    os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
    recreate_index()
    print("\nNow run: python scripts/ingest.py")
