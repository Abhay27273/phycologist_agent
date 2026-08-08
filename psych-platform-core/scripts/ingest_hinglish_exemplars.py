"""
Hinglish (hi-en-codeswitch) style exemplar ingestion.

Source: 4 self-authored/adapted session transcripts (Tom, Lucy, Neha, Sam) —
NOT expert-annotated like AnnoMI. technique_label and emotion_category arrive
empty in the source CSVs, so this script auto-labels each Psychologist turn
via a single Gemini call per session (Gemini, not Groq, because this is an
offline batch job where Gemini's slower "thinking phase" latency doesn't
matter, and it sidesteps Groq's daily token quota entirely).

Every exemplar from this script is tagged source="hinglish_self_authored" and
label_confidence="machine_labeled" — explicitly lower confidence than
AnnoMI's expert annotations (source="AnnoMI"). Retrieval doesn't currently
distinguish on this field, but it's preserved for future filtering/audit if
machine-labeled exemplars ever need to be excluded or reviewed separately.

CONTENT FILTER: some Psychologist turns are specialized safeguarding/forensic
screening questions (e.g. Sam's abuse-screening questions) rather than
generalizable therapeutic technique. Those aren't generic "how to ask an open
question" templates — they require the full clinical/legal context they were
asked in, and decontextualized retrieval could surface a forensic screening
question as generic style guidance in an unrelated conversation. The labeling
prompt asks Gemini to flag these explicitly (exclude_reason) and they are
skipped rather than ingested, regardless of what technique they'd otherwise
map to.

risk_flag from the source CSV is carried through into exemplar metadata
unchanged for audit traceability — it does not gate ingestion (per explicit
instruction: review_required_* is informational, not a block).

Usage:
    python scripts/ingest_hinglish_exemplars.py --dry-run
    python scripts/ingest_hinglish_exemplars.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "hinglish_sessions"


def _load_session(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["turn_id"]))
    return rows


def _estimate_turn_position(turn_idx: int, total_turns: int) -> str:
    if total_turns == 0:
        return "mid"
    pct = turn_idx / total_turns
    if pct < 0.2:
        return "opening"
    if pct > 0.8:
        return "closing"
    return "mid"


def _build_labeling_prompt(session_id: str, rows: list[dict]) -> str:
    from app.services.therapeutic_prompt import MOVE_SET, _MOVE_INSTRUCTIONS_EN
    from app.graph.nodes.sentiment import _VALID_MOODS

    move_lines = "\n".join(f"- {m}: {_MOVE_INSTRUCTIONS_EN[m]}" for m in sorted(MOVE_SET))
    mood_list = ", ".join(sorted(_VALID_MOODS))

    transcript_lines = "\n".join(
        f"[{r['turn_id']}] {r['speaker']}: {r['dialogue']}" for r in rows
    )

    return f"""You are labeling a therapy session transcript (Hindi-English codeswitched) for a
technique-tagged retrieval bank. Only label PSYCHOLOGIST turns — skip Patient turns entirely.

Our therapeutic move taxonomy (a turn must cleanly match ONE of these to get a technique_label
— if it's a closed/forensic/structured-intake question or doesn't clearly fit ANY of these,
use "none" rather than forcing a mismatch):
{move_lines}

Mood taxonomy for emotion_category (label the CLIENT's emotional state around that point in the
conversation, using the turn just before/around the Psychologist's turn as evidence): {mood_list}

IMPORTANT — exclude_reason: if a Psychologist turn is a SPECIALIZED forensic/safeguarding
screening question (e.g. asking specifically about physical/sexual abuse, self-harm method
details, or other structured child-protection/legal-intake protocol questions) rather than
general therapeutic technique, set exclude_reason to a short phrase describing why (e.g.
"specialized_safeguarding_screening"). Such turns should still get their best-fit technique_label
if one applies, but the exclude_reason marks them as unsuitable for decontextualized reuse as a
generic style template elsewhere.

Session: {session_id}

Transcript:
{transcript_lines}

Return ONLY a JSON array (no markdown fences, no commentary), one object per PSYCHOLOGIST turn:
[{{"turn_id": "<id>", "technique_label": "<move_name or none>", "emotion_category": "<mood or neutral>", "affect_valence": "<low|neutral|high>", "exclude_reason": "<string or null>"}}]
"""


async def _label_session(llm, session_id: str, rows: list[dict]) -> dict[str, dict]:
    prompt = _build_labeling_prompt(session_id, rows)
    response = await llm.ainvoke(prompt)
    content = response.content.replace("```json", "").replace("```", "").strip()
    labels = json.loads(content)
    return {str(item["turn_id"]): item for item in labels}


def _build_exemplars(session_id: str, rows: list[dict], labels: dict[str, dict]) -> list[dict[str, Any]]:
    from app.services.therapeutic_prompt import MOVE_SET

    exemplars = []
    psych_rows = [r for r in rows if r["speaker"] == "Psychologist"]
    total = len(psych_rows)
    session_risk_flag = next((r["risk_flag"] for r in rows if r.get("risk_flag")), "")

    for idx, row in enumerate(psych_rows):
        label = labels.get(row["turn_id"])
        if label is None:
            logger.warning("No label returned for %s turn %s — skipping", session_id, row["turn_id"])
            continue

        move = label.get("technique_label")
        if move not in MOVE_SET:
            continue  # "none" or hallucinated move name — skip, don't guess

        if label.get("exclude_reason"):
            logger.info(
                "Skipping %s turn %s (excluded: %s): %.60s",
                session_id, row["turn_id"], label["exclude_reason"], row["dialogue"],
            )
            continue

        text = row["dialogue"].strip()
        if len(text) < 15:
            continue

        # Preceding patient turn, for logged context (not used for valence —
        # Gemini already judged valence with the full transcript in view).
        client_context = ""
        row_idx = rows.index(row)
        for preceding in reversed(rows[:row_idx]):
            if preceding["speaker"] == "Patient":
                client_context = preceding["dialogue"].strip()[:200]
                break

        exemplars.append({
            "text": text,
            "move": move,
            "affect_valence": label.get("affect_valence", "neutral"),
            "emotion_category": label.get("emotion_category", "neutral"),
            "turn_position": _estimate_turn_position(idx, total),
            "source": "hinglish_self_authored",
            "label_confidence": "machine_labeled",
            "transcript_id": session_id,
            "language": "hinglish",
            "register": "hinglish-casual",
            "client_context_snippet": client_context,
            "session_risk_flag": session_risk_flag,
        })

    return exemplars


def _ingest_into_qdrant(exemplars: list[dict[str, Any]]) -> None:
    """Add to the EXISTING psych-style collection — never recreate it here,
    or the 2176 AnnoMI exemplars already ingested would be destroyed."""
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore
    from app.core.config import settings

    docs = [Document(page_content=ex["text"], metadata={k: v for k, v in ex.items() if k != "text"})
            for ex in exemplars]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    if settings.QDRANT_MODE == "server":
        client = QdrantClient(url=settings.QDRANT_URL)
    else:
        project_root = Path(__file__).resolve().parent.parent
        client = QdrantClient(path=str(project_root / settings.QDRANT_PATH.lstrip("./")))

    if not client.collection_exists(settings.QDRANT_STYLE_COLLECTION):
        raise RuntimeError(
            f"Collection '{settings.QDRANT_STYLE_COLLECTION}' does not exist — "
            "run scripts/ingest_annomi.py first to create it."
        )

    store = QdrantVectorStore(client=client, collection_name=settings.QDRANT_STYLE_COLLECTION, embedding=embeddings)
    store.add_documents(docs)
    logger.info("Added %d Hinglish exemplars to '%s'.", len(docs), settings.QDRANT_STYLE_COLLECTION)


async def main_async(dry_run: bool) -> None:
    from app.core.config import settings
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL, google_api_key=settings.GOOGLE_API_KEY, temperature=0)

    csv_paths = sorted(SESSIONS_DIR.glob("*.csv"))
    if not csv_paths:
        logger.error("No CSVs found in %s", SESSIONS_DIR)
        sys.exit(1)

    all_exemplars: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        rows = _load_session(csv_path)
        session_id = rows[0]["session_id"]
        logger.info("Labeling session %s (%d turns)...", session_id, len(rows))
        labels = await _label_session(llm, session_id, rows)
        exemplars = _build_exemplars(session_id, rows, labels)
        logger.info("Session %s -> %d usable exemplars (of %d Psychologist turns)",
                    session_id, len(exemplars), sum(1 for r in rows if r["speaker"] == "Psychologist"))
        all_exemplars.extend(exemplars)

    from collections import Counter
    print("\nMove distribution (Hinglish set):")
    for m, n in Counter(e["move"] for e in all_exemplars).most_common():
        print(f"  {m:<25} {n:>3}")
    print("\nBy session:")
    for sid, n in Counter(e["transcript_id"] for e in all_exemplars).most_common():
        print(f"  {sid:<10} {n:>3}")

    if dry_run:
        print("\nDRY RUN — sample exemplars:")
        for ex in all_exemplars[:8]:
            print(f"  [{ex['transcript_id']}] move={ex['move']:<20} valence={ex['affect_valence']:<8} "
                  f"excl={ex.get('session_risk_flag') or '-'} | {ex['text'][:70]}")
        return

    _ingest_into_qdrant(all_exemplars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.dry_run))


if __name__ == "__main__":
    main()
