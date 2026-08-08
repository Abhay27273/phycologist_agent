"""
MemoryService — Phase 4.

Three-store memory architecture (§3.2):
  1. Facts / entities  → UserFact table (temporal validity, never-overwrite)
  2. Narrative         → ChatSession.summary (rolling prose — unchanged)
  3. Trajectory        → MoodTrajectory table (numeric time-series)

Recall gating (§3.3):
  Default is silence — most turns inject NO memory context.
  Recall fires only when explicit trigger conditions are met.
  "A system that opens every reply with a callback to a prior session is not
  attentive, it's performing attentiveness."

DPDP compliance:
  forget(fact_id) is a first-class primitive — hard delete per §3.3.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.models import MoodTrajectory, UserFact

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-end consolidation (structured entities + long-term dialogue memory)
# ---------------------------------------------------------------------------
#
# Extends the existing UserFact categories (person | relationship | event |
# commitment | preference — free-text column, no schema change needed) with
# four clinically-oriented ones. Distinct from the rolling SummaryNode prose
# summary: that's a flat narrative for context; this is structured enough to
# query ("what are this user's active triggers") and paired with a separate
# semantic-memory write (see rag_service.store_patient_memory) for recall
# that isn't just "the 3 most recent facts" (see build_memory_context above).

_ENTITY_CATEGORIES = {"trigger", "coping_strategy", "core_belief", "goal"}

_CONSOLIDATION_PROMPT_TEMPLATE = """\
Analyze this therapy conversation and extract structured insights.

Conversation:
{transcript}

Client's current mood: {mood}

Return a JSON object with EXACTLY these keys:
  "anxiety_score" — integer 1-10, overall distress level shown across this conversation.
  "valence_score" — float -1.0 to 1.0 (-1 very negative, 0 neutral, 1 very positive).
  "entities" — array of objects, each: {{"category": one of [trigger, coping_strategy,
               core_belief, goal], "description": short clinical description,
               "entity_label": short label for matching in later conversations (e.g. a
               person's name, a situation like "team meetings", or null)}}.
               Only include entities with clear, specific evidence in THIS conversation —
               do not invent or infer beyond what was actually said. Empty array if none.
  "highlights" — array of 1-3 short strings (1 sentence each), each a specific moment or
               disclosure worth remembering in future sessions (not a generic summary —
               the kind of thing a real clinician would jot down as a note to self).
               Empty array if nothing distinctive.

Return ONLY valid JSON, no markdown, no explanation.
"""


def _build_transcript(history: list) -> str:
    lines = []
    for m in history:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            label = "Client" if m["role"] == "user" else "Therapist"
            lines.append(f"{label}: {m.get('content', '')}")
        elif hasattr(m, "type"):
            label = "Client" if getattr(m, "type", "") == "human" else "Therapist"
            lines.append(f"{label}: {getattr(m, 'content', '')}")
    return "\n".join(lines[-40:])  # cap — this is a distinct, deeper pass than SummaryNode's rolling one


async def extract_session_insights(llm_service, history: list, mood: str) -> dict[str, Any]:
    """
    One LLM call, reusing the existing analyze_sentiment_raw(prompt) -> dict
    interface every LLMService implementation already has (built for
    SentimentNode's rich prompt, generic enough to reuse here without adding
    a new interface method). Returns {} on any failure — callers should treat
    that as "nothing to consolidate this time", not an error to surface.
    """
    transcript = _build_transcript(history)
    if not transcript.strip():
        return {}
    prompt = _CONSOLIDATION_PROMPT_TEMPLATE.format(transcript=transcript, mood=mood)
    try:
        result = await llm_service.analyze_sentiment_raw(prompt)
        if not isinstance(result, dict):
            return {}
        return result
    except Exception as e:
        logger.error("extract_session_insights failed: %s", e)
        return {}

# ---------------------------------------------------------------------------
# Recall trigger conditions
# ---------------------------------------------------------------------------

# Phrases that signal the user is referencing the past
_PAST_REFERENCE_RE = re.compile(
    r"\b(last time|before|earlier|remember when|you said|i told you|we talked|"
    r"pehle|jo baat|usne|woh waqt|yaad hai|tumhe pata hai)\b",
    re.IGNORECASE,
)

# Exclusive-reliance phrases (dependency monitoring §5.4)
_EXCLUSIVE_RELIANCE_RE = re.compile(
    r"\b(only one who (understands|gets me|listens)|"
    r"don't need anyone else|"
    r"you're all i (have|need)|"
    r"sirf tum|bas tum|koi aur nahi samjhega)\b",
    re.IGNORECASE,
)

# Human-support mentions
_HUMAN_SUPPORT_RE = re.compile(
    r"\b(friend|family|therapist|counsellor|doctor|family member|"
    r"dost|yaar|parivaar|ghar wale|psychologist)\b",
    re.IGNORECASE,
)


def _user_references_past(message: str) -> bool:
    return bool(_PAST_REFERENCE_RE.search(message))


def _mentions_exclusive_reliance(message: str) -> bool:
    return bool(_EXCLUSIVE_RELIANCE_RE.search(message))


def _mentions_human_support(message: str) -> bool:
    return bool(_HUMAN_SUPPORT_RE.search(message))


# ---------------------------------------------------------------------------
# MemoryService
# ---------------------------------------------------------------------------

class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # -----------------------------------------------------------------------
    # Trajectory store
    # -----------------------------------------------------------------------

    async def record_turn(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        mood: str,
        risk_score: int,
    ) -> None:
        """Persist one turn's mood + risk_score to the trajectory table."""
        try:
            row = MoodTrajectory(
                user_id=user_id,
                session_id=session_id,
                turn_index=turn_index,
                mood=mood,
                risk_score=risk_score,
                recorded_at=datetime.now(timezone.utc),
            )
            self.db.add(row)
            await self.db.commit()
        except Exception as e:
            logger.error("MoodTrajectory record_turn failed: %s", e)

    async def get_recent_trajectory(
        self,
        user_id: str,
        last_n_sessions: int = 5,
    ) -> list[dict]:
        """Return recent mood/risk rows for trajectory slope detection."""
        try:
            result = await self.db.execute(
                select(MoodTrajectory)
                .where(MoodTrajectory.user_id == user_id)
                .order_by(MoodTrajectory.recorded_at.desc())
                .limit(last_n_sessions * 20)  # ~20 turns per session
            )
            rows = result.scalars().all()
            return [
                {
                    "mood": r.mood,
                    "risk_score": r.risk_score,
                    "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
                }
                for r in reversed(rows)
            ]
        except Exception as e:
            logger.error("get_recent_trajectory failed: %s", e)
            return []

    async def risk_slope_exceeds(self, user_id: str, threshold: float = 2.0) -> bool:
        """Return True if average risk_score is rising > threshold over last 3 sessions."""
        trajectory = await self.get_recent_trajectory(user_id, last_n_sessions=3)
        if len(trajectory) < 6:
            return False
        mid = len(trajectory) // 2
        early_avg = sum(t["risk_score"] for t in trajectory[:mid]) / mid
        late_avg = sum(t["risk_score"] for t in trajectory[mid:]) / (len(trajectory) - mid)
        return (late_avg - early_avg) > threshold

    # -----------------------------------------------------------------------
    # Fact store
    # -----------------------------------------------------------------------

    async def store_fact(
        self,
        user_id: str,
        fact_text: str,
        entity_label: str | None = None,
        category: str = "general",
        source_session_id: str | None = None,
        confidence: float = 1.0,
    ) -> int:
        """Insert a new fact. Never overwrites — invalidate old facts separately."""
        try:
            row = UserFact(
                user_id=user_id,
                fact_text=fact_text,
                entity_label=entity_label,
                category=category,
                source_session_id=source_session_id,
                confidence=confidence,
            )
            self.db.add(row)
            await self.db.commit()
            await self.db.refresh(row)
            return row.id
        except Exception as e:
            logger.error("store_fact failed: %s", e)
            return -1

    async def get_active_facts(self, user_id: str) -> list[dict]:
        """Return all currently valid facts for a user."""
        try:
            now = datetime.now(timezone.utc)
            result = await self.db.execute(
                select(UserFact).where(
                    and_(
                        UserFact.user_id == user_id,
                        or_(
                            UserFact.validity_end.is_(None),
                            UserFact.validity_end > now,
                        ),
                    )
                ).order_by(UserFact.created_at.desc())
            )
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "fact_text": r.fact_text,
                    "entity_label": r.entity_label,
                    "category": r.category,
                    "confidence": r.confidence,
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("get_active_facts failed: %s", e)
            return []

    async def forget(self, user_id: str, fact_id: int) -> bool:
        """
        Hard delete a fact by ID — DPDP erasure right (§3.3 / §4.5).
        Verifies user_id ownership before deleting.
        """
        try:
            result = await self.db.execute(
                select(UserFact).where(
                    and_(UserFact.id == fact_id, UserFact.user_id == user_id)
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return False
            await self.db.delete(row)
            await self.db.commit()
            logger.info("UserFact %d deleted for user %s (forget request)", fact_id, user_id)
            return True
        except Exception as e:
            logger.error("forget(%d) failed: %s", fact_id, e)
            return False

    async def invalidate_fact(self, user_id: str, fact_id: int) -> bool:
        """Soft-invalidate a fact (set validity_end = now) — keeps audit trail."""
        try:
            result = await self.db.execute(
                select(UserFact).where(
                    and_(UserFact.id == fact_id, UserFact.user_id == user_id)
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return False
            row.validity_end = datetime.now(timezone.utc)
            await self.db.commit()
            return True
        except Exception as e:
            logger.error("invalidate_fact(%d) failed: %s", fact_id, e)
            return False

    # -----------------------------------------------------------------------
    # Session-end consolidation persistence
    # -----------------------------------------------------------------------

    async def apply_consolidation(
        self,
        user_id: str,
        session_id: str,
        insights: dict[str, Any],
        rag_service,
    ) -> None:
        """
        Persist the output of extract_session_insights(): entities to
        UserFact (relational, queryable), highlights to the patient_memory
        Qdrant collection (semantic recall). Best-effort — logs and continues
        past a bad entity or a Qdrant hiccup rather than losing the whole batch.
        """
        for entity in insights.get("entities") or []:
            category = entity.get("category")
            description = (entity.get("description") or "").strip()
            if category not in _ENTITY_CATEGORIES or not description:
                continue
            await self.store_fact(
                user_id=user_id,
                fact_text=description,
                entity_label=entity.get("entity_label"),
                category=category,
                source_session_id=session_id,
            )

        for highlight in insights.get("highlights") or []:
            text = highlight.strip() if isinstance(highlight, str) else ""
            if text:
                await rag_service.store_patient_memory(
                    user_id=user_id, session_id=session_id, content=text, chunk_type="dialogue_highlight",
                )

    # -----------------------------------------------------------------------
    # Recall gating (§3.3)
    # -----------------------------------------------------------------------

    async def should_recall(
        self,
        user_id: str,
        session_id: str,
        message: str,
        days_since_last_session: int = 0,
        mood: str = "neutral",
    ) -> bool:
        """
        Gating function — returns True only when a recall trigger fires.
        Default is False. Silence about the past is the correct default.
        """
        # Trigger 1: user explicitly references the past
        if _user_references_past(message):
            logger.debug("Recall trigger: user_references_past")
            return True

        # Trigger 2: entity overlap with stored facts
        facts = await self.get_active_facts(user_id)
        if facts:
            message_lower = message.lower()
            for fact in facts:
                label = (fact.get("entity_label") or "").lower()
                if label and len(label) > 2 and label in message_lower:
                    logger.debug("Recall trigger: entity_overlap '%s'", label)
                    return True

        # Trigger 3: long absence (re-opening earns one orienting callback)
        if days_since_last_session >= 7:
            logger.debug("Recall trigger: long_absence %d days", days_since_last_session)
            return True

        # Trigger 4: trajectory slope — risk rising
        if await self.risk_slope_exceeds(user_id):
            logger.debug("Recall trigger: trajectory_slope_exceeds")
            return True

        return False

    async def build_memory_context(
        self,
        user_id: str,
        message: str,
        days_since_last_session: int = 0,
        mood: str = "neutral",
        session_id: str = "",
    ) -> str:
        """
        Build a concise memory context string to inject into the prompt,
        but ONLY when a recall trigger fires. Returns empty string otherwise.
        """
        if not await self.should_recall(user_id, session_id, message, days_since_last_session, mood):
            return ""

        facts = await self.get_active_facts(user_id)
        if not facts:
            return ""

        # Return only the 3 most recent high-confidence facts — not a dossier
        top_facts = sorted(facts, key=lambda f: f.get("confidence", 0), reverse=True)[:3]
        lines = [f["fact_text"] for f in top_facts]
        return "[What I remember]\n" + "\n".join(f"- {l}" for l in lines)

    # -----------------------------------------------------------------------
    # Dependency signal helpers
    # -----------------------------------------------------------------------

    def detect_exclusive_reliance(self, message: str) -> bool:
        return _mentions_exclusive_reliance(message)

    def detect_human_support_mention(self, message: str) -> bool:
        return _mentions_human_support(message)
