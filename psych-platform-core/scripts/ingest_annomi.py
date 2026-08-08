"""
AnnoMI style exemplar ingestion — Phase 2.2.

Ingests therapist turn transcripts from AnnoMI into the style_exemplars vector
collection (QDRANT_STYLE_COLLECTION / PINECONE_STYLE_INDEX_NAME), keyed on
therapist move + client state metadata.

WHY a separate collection: clinical textbook chunks (clinical_kb) teach what
is true; transcript exemplars (style_exemplars) teach how a turn is shaped.
Mixed into the same collection they collapse — the model quotes clinical content
conversationally, or treats transcripts as clinical evidence. See §1.2.

RETRIEVAL NOTE: exemplars are retrieved by metadata filter (move + valence),
NOT by semantic similarity. Semantic similarity retrieves someone else's specific
situation, and the model echoes its particulars. See §1.3.

SCHEMA NOTE: this parses AnnoMI-full.csv's REAL columns, verified directly
against https://github.com/uccollab/AnnoMI (fetched 2026-07-31). The dataset
does NOT have a single per-utterance "behaviour code" string — it has
`main_therapist_behaviour` (n/a|reflection|question|therapist_input|other) plus
separate subtype columns (`reflection_subtype`, `question_subtype`,
`therapist_input_subtype`) that only apply when the parent behaviour matches.
There is no code at all for affirmation, normalising, summarising, or
sit_with_it/silence — AnnoMI's annotation schema doesn't distinguish those,
so this script can only ever populate simple_reflection, complex_reflection,
open_question, and psychoeducation. Utterances coded "other" are skipped
rather than guessed at, to keep the exemplar bank technique-pure.

Rows are per-annotator (up to 10 annotators per utterance, most utterances
have exactly 1). This script groups by (transcript_id, utterance_id) and
takes the majority vote across annotators; ties are skipped as low-confidence.

Every exemplar is filtered to mi_quality == "high" by default — AnnoMI
deliberately includes low-quality/poor-MI demonstrations for fidelity
research, and those must never be served as style guidance to real users.

Usage:
    # Download AnnoMI-full.csv first:
    #   curl --ssl-no-revoke -sL -o data/annomi_full.csv \\
    #     https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-full.csv
    #   (the --ssl-no-revoke flag works around a schannel CRL-check failure
    #   seen on this network; drop it if your environment doesn't need it)
    #   Paper: https://www.mdpi.com/1999-5903/15/3/110

    python scripts/ingest_annomi.py --csv data/annomi_full.csv
    python scripts/ingest_annomi.py --csv data/annomi_full.csv --backend qdrant
    python scripts/ingest_annomi.py --csv data/annomi_full.csv --dry-run   # preview only

    # For Hinglish/Hindi exemplars (no open dataset exists yet — see
    # architecture_technique_retrieval memory — this is future scaffolding):
    python scripts/ingest_annomi.py --csv data/hinglish_exemplars.csv --language hinglish --register hinglish-casual
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AnnoMI behaviour code → our move taxonomy mapping
# ---------------------------------------------------------------------------
#
# Our 9-move taxonomy is intentionally narrower than AnnoMI's coding scheme,
# and AnnoMI's scheme doesn't cover all 9 moves either. Only include a mapping
# here if AnnoMI's own annotation gives high-confidence evidence it's that
# exact technique — do not guess at "other"-coded utterances.

REFLECTION_SUBTYPE_TO_MOVE = {
    "simple": "simple_reflection",
    "complex": "complex_reflection",
}
QUESTION_SUBTYPE_TO_MOVE = {
    "open": "open_question",
    # "closed" intentionally has no mapping — not in our taxonomy.
}
THERAPIST_INPUT_SUBTYPE_TO_MOVE = {
    "information": "psychoeducation",
    # advice / negotiation / options intentionally unmapped — advice-giving is
    # a different therapeutic function than grounded psychoeducation, and
    # conflating them would poison the psychoeducation exemplar bank.
}

# AnnoMI client talk types (for retrieval context)
CLIENT_TALK_MAP: dict[str, str] = {
    "change": "change_talk",
    "sustain": "sustain_talk",
    "neutral": "neutral",
}


def _map_affect_valence(client_text: str) -> str:
    """
    Rough valence from the preceding CLIENT utterance — AnnoMI doesn't
    annotate valence directly, and we want the exemplar keyed on what the
    client was feeling (matching how TherapyNode queries by current_mood),
    not on the therapist's own phrasing.
    """
    text = (client_text or "").strip().lower()
    if any(w in text for w in ["pain", "hurt", "sad", "angry", "scared", "hopeless", "worthless", "alone"]):
        return "low"
    if any(w in text for w in ["better", "hopeful", "trying", "good", "change", "want"]):
        return "high"
    return "neutral"


def _majority(values: list[str]) -> str | None:
    """Majority vote among non-'n/a' values; None on tie or no signal (skip — low confidence)."""
    filtered = [v for v in values if v and v != "n/a"]
    if not filtered:
        return None
    counts = Counter(filtered)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None  # tie — annotators disagreed, skip rather than guess
    return top[0][0]


def _estimate_turn_position(turn_idx: int, total_turns: int) -> str:
    if total_turns == 0:
        return "mid"
    pct = turn_idx / total_turns
    if pct < 0.2:
        return "opening"
    if pct > 0.8:
        return "closing"
    return "mid"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_annomi_csv(
    csv_path: Path,
    language: str = "en",
    register: str = "en",
    min_quality: str = "high",
) -> list[dict[str, Any]]:
    """
    Parse AnnoMI-full.csv into a list of exemplar dicts ready for ingestion.

    Real columns (verified against https://github.com/uccollab/AnnoMI):
        mi_quality, transcript_id, video_title, video_url, topic, utterance_id,
        interlocutor, timestamp, utterance_text, annotator_id,
        therapist_input_exists, therapist_input_subtype, reflection_exists,
        reflection_subtype, question_exists, question_subtype,
        main_therapist_behaviour, client_talk_type

    Rows are per-annotator (most utterances have exactly 1 row; a 428-utterance
    inter-annotator-agreement subsample has 10). This groups by
    (transcript_id, utterance_id) and majority-votes across annotators before
    mapping to our move taxonomy — skips ties as low-confidence.

    Returns only therapist turns with a known, high-confidence move mapping,
    filtered to mi_quality >= min_quality (default: high-quality demonstrations
    only — AnnoMI's low-quality rows are deliberately poor MI and must never
    be served as style guidance).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if min_quality == "high":
        rows = [r for r in rows if r.get("mi_quality") == "high"]

    # Group per-annotator rows into one row per unique utterance.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["transcript_id"], row["utterance_id"])].append(row)

    # Rebuild per-transcript ordered turn sequences (one row per utterance)
    # so we can find each therapist turn's preceding client turn for context
    # and its turn position within that transcript.
    transcripts: dict[str, list[dict]] = defaultdict(list)
    for (transcript_id, utterance_id), grp in groups.items():
        base = grp[0]
        merged = dict(base)
        merged["main_therapist_behaviour"] = _majority([g["main_therapist_behaviour"] for g in grp])
        merged["reflection_subtype"] = _majority([g["reflection_subtype"] for g in grp])
        merged["question_subtype"] = _majority([g["question_subtype"] for g in grp])
        merged["therapist_input_subtype"] = _majority([g["therapist_input_subtype"] for g in grp])
        merged["client_talk_type"] = _majority([g["client_talk_type"] for g in grp])
        transcripts[transcript_id].append(merged)

    for turns in transcripts.values():
        turns.sort(key=lambda r: int(r["utterance_id"]))

    exemplars: list[dict[str, Any]] = []
    skipped_other = 0
    skipped_tie_or_unmapped = 0

    for transcript_id, turns in transcripts.items():
        therapist_turns = [t for t in turns if t["interlocutor"] == "therapist"]
        total = len(therapist_turns)

        for t_idx, row in enumerate(therapist_turns):
            text = (row.get("utterance_text") or "").strip()
            if not text or len(text) < 20:
                continue

            behaviour = row.get("main_therapist_behaviour")
            move = None
            if behaviour == "reflection":
                move = REFLECTION_SUBTYPE_TO_MOVE.get(row.get("reflection_subtype") or "")
            elif behaviour == "question":
                move = QUESTION_SUBTYPE_TO_MOVE.get(row.get("question_subtype") or "")
            elif behaviour == "therapist_input":
                move = THERAPIST_INPUT_SUBTYPE_TO_MOVE.get(row.get("therapist_input_subtype") or "")
            elif behaviour == "other":
                skipped_other += 1
                continue

            if move is None:
                skipped_tie_or_unmapped += 1
                continue

            client_talk = CLIENT_TALK_MAP.get(row.get("client_talk_type") or "", "neutral")
            turn_position = _estimate_turn_position(t_idx, total)

            # Find the preceding client turn for context + valence tagging.
            client_context = ""
            row_idx = turns.index(row)
            for preceding in reversed(turns[:row_idx]):
                if preceding["interlocutor"] == "client":
                    client_context = (preceding.get("utterance_text") or "").strip()[:200]
                    break
            affect_valence = _map_affect_valence(client_context)

            exemplars.append({
                "text": text,
                "move": move,
                "client_talk_type": client_talk,
                "affect_valence": affect_valence,
                "turn_position": turn_position,
                "source": "AnnoMI",
                "transcript_id": transcript_id,
                "language": language,
                "register": register,
                "client_context_snippet": client_context,
            })

    logger.info(
        "Parsed %d exemplars from %d transcripts (skipped %d 'other'-coded, "
        "%d tied/unmapped subtype)",
        len(exemplars), len(transcripts), skipped_other, skipped_tie_or_unmapped,
    )
    return exemplars


# ---------------------------------------------------------------------------
# Vector store ingestion
# ---------------------------------------------------------------------------

def ingest_exemplars(
    exemplars: list[dict[str, Any]],
    backend: str,
    dry_run: bool = False,
) -> None:
    """Ingest exemplar dicts into the style_exemplars vector collection."""
    from langchain_core.documents import Document

    docs = []
    for ex in exemplars:
        metadata = {k: v for k, v in ex.items() if k != "text"}
        docs.append(Document(page_content=ex["text"], metadata=metadata))

    logger.info("Prepared %d documents for ingestion", len(docs))

    if dry_run:
        logger.info("DRY RUN — first 5 exemplars:")
        for d in docs[:5]:
            logger.info("  move=%-22s valence=%-8s | %s", d.metadata["move"], d.metadata["affect_valence"], d.page_content[:80])
        return

    from langchain_huggingface import HuggingFaceEmbeddings
    from app.core.config import settings

    embed_model = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    if backend == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore

        if settings.QDRANT_MODE == "server":
            client = QdrantClient(url=settings.QDRANT_URL)
        else:
            project_root = Path(__file__).resolve().parent.parent
            qdrant_path = str(project_root / settings.QDRANT_PATH.lstrip("./"))
            client = QdrantClient(path=qdrant_path)

        # QdrantVectorStore.from_documents() builds its OWN client from
        # connection kwargs — passing an already-constructed client object
        # gets misrouted into httpx.Client(**kwargs) and crashes. Create the
        # collection explicitly, then reuse the constructor pattern that
        # rag_service.py's _build_vector_store already uses successfully.
        vector_size = len(embeddings.embed_query("dimension probe"))
        client.recreate_collection(
            collection_name=settings.QDRANT_STYLE_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_STYLE_COLLECTION,
            embedding=embeddings,
        )

        # Batch ingest in chunks of 100
        BATCH = 100
        for i in range(0, len(docs), BATCH):
            batch = docs[i:i + BATCH]
            store.add_documents(batch)
            logger.info("Ingested batch %d/%d", i + BATCH, len(docs))

    elif backend == "pinecone":
        os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        from langchain_pinecone import PineconeVectorStore

        BATCH = 100
        for i in range(0, len(docs), BATCH):
            batch = docs[i:i + BATCH]
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=settings.PINECONE_STYLE_INDEX_NAME,
            )
            logger.info("Ingested batch %d/%d", i + BATCH, len(docs))

    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.info("Done. %d exemplars ingested into '%s' collection.", len(docs),
                settings.QDRANT_STYLE_COLLECTION if backend == "qdrant" else settings.PINECONE_STYLE_INDEX_NAME)


# ---------------------------------------------------------------------------
# Move distribution report
# ---------------------------------------------------------------------------

def print_distribution(exemplars: list[dict]) -> None:
    from collections import Counter
    moves = Counter(e["move"] for e in exemplars)
    valences = Counter(e["affect_valence"] for e in exemplars)
    positions = Counter(e["turn_position"] for e in exemplars)
    print("\nMove distribution:")
    for m, n in sorted(moves.items(), key=lambda x: -x[1]):
        print(f"  {m:<25} {n:>5}")
    print("\nAffect valence distribution:")
    for v, n in valences.most_common():
        print(f"  {v:<15} {n:>5}")
    print("\nTurn position distribution:")
    for p, n in positions.most_common():
        print(f"  {p:<15} {n:>5}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import os

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest AnnoMI style exemplars into the vector store")
    parser.add_argument("--csv", required=True, help="Path to AnnoMI CSV file")
    parser.add_argument("--backend", default=None, help="qdrant|pinecone (default: from VECTOR_DB_BACKEND env)")
    parser.add_argument("--language", default="en", help="Language code: en | hinglish (default: en)")
    parser.add_argument("--register", default="en", help="Register tag: en | hinglish-casual | formal-hi (default: en)")
    parser.add_argument(
        "--include-low-quality", action="store_true",
        help="Include mi_quality=low (deliberately poor MI demonstrations) — off by default, "
             "never enable this for a collection served to real users",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and preview without writing to vector store")
    parser.add_argument("--stats", action="store_true", help="Print move distribution and exit")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    min_quality = "all" if args.include_low_quality else "high"
    exemplars = parse_annomi_csv(csv_path, language=args.language, register=args.register, min_quality=min_quality)

    if args.stats or args.dry_run:
        print_distribution(exemplars)

    if args.stats:
        return

    from app.core.config import settings
    backend = args.backend or settings.VECTOR_DB_BACKEND
    ingest_exemplars(exemplars, backend=backend, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
