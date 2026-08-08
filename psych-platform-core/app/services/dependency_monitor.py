"""
Dependency monitoring — Phase 5.2.

Weekly signals (§5.4):
  1. session_frequency_trend  — sessions/week rising over last 4 weeks
  2. night_time_turn_share    — 00:00–05:00 concentration of all turns
  3. exclusive_reliance_count — "you're my only..." phrases this week
  4. human_support_mention_count — declining mentions of friends/therapist/family
  5. signal_level             — composite: "normal" | "moderate" | "high"

Graded responses:
  normal   → no action
  moderate → gentle prompt toward human support woven into next response
  high     → structured suggestion ("I care about you having other support...") +
             flag for operator review if signal persists across two consecutive weeks

Design principle: no single turn looks wrong — only the trend does.
This module operates on stored trajectory + dependency_signal rows, not on
live messages. The caller (e.g. a nightly celery task or the chat route
post-turn hook) is responsible for scheduling compute.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.models import (
    ChatMessage,
    ChatSession,
    DependencySignal,
    MoodTrajectory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_SESSION_FREQ_MODERATE = 7     # sessions/week — concern starts here
_SESSION_FREQ_HIGH = 14        # ~twice daily
_NIGHT_SHARE_MODERATE = 0.30   # 30 % of turns between 00:00–05:00
_NIGHT_SHARE_HIGH = 0.50
_EXCLUSIVE_MODERATE = 3        # exclusive-reliance phrase count/week
_EXCLUSIVE_HIGH = 6
_HUMAN_SUPPORT_DECLINE_WEEKS = 2  # human-support count drops two weeks running → concern


def _week_start(dt: datetime) -> datetime:
    """Return the Monday 00:00 UTC of the week containing *dt*."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _is_night_turn(ts: Optional[datetime]) -> bool:
    if ts is None:
        return False
    utc_ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
    return 0 <= utc_ts.hour < 5


def _composite_level(
    session_count: int,
    night_share: float,
    exclusive_count: int,
    human_support_declining: bool,
) -> str:
    high_flags = 0
    moderate_flags = 0

    if session_count >= _SESSION_FREQ_HIGH:
        high_flags += 1
    elif session_count >= _SESSION_FREQ_MODERATE:
        moderate_flags += 1

    if night_share >= _NIGHT_SHARE_HIGH:
        high_flags += 1
    elif night_share >= _NIGHT_SHARE_MODERATE:
        moderate_flags += 1

    if exclusive_count >= _EXCLUSIVE_HIGH:
        high_flags += 1
    elif exclusive_count >= _EXCLUSIVE_MODERATE:
        moderate_flags += 1

    if human_support_declining:
        moderate_flags += 1

    if high_flags >= 1:
        return "high"
    if moderate_flags >= 2:
        return "moderate"
    return "normal"


# ---------------------------------------------------------------------------
# DependencyMonitor
# ---------------------------------------------------------------------------

class DependencyMonitor:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def compute_weekly_signal(
        self,
        user_id: str,
        week_start: Optional[datetime] = None,
    ) -> DependencySignal:
        """
        Compute and persist the dependency signal for *user_id* for the week
        starting at *week_start* (defaults to the current week).
        """
        if week_start is None:
            week_start = _week_start(datetime.now(timezone.utc))
        week_end = week_start + timedelta(weeks=1)

        # ----- Session count for this week -----
        session_result = await self.db.execute(
            select(func.count(ChatSession.id)).where(
                and_(
                    ChatSession.user_id == user_id,
                    ChatSession.created_at >= week_start,
                    ChatSession.created_at < week_end,
                )
            )
        )
        session_count: int = session_result.scalar_one_or_none() or 0

        # ----- Turn counts + night-time share -----
        msg_result = await self.db.execute(
            select(ChatMessage.timestamp).where(
                and_(
                    ChatMessage.session_id.in_(
                        select(ChatSession.id).where(
                            and_(
                                ChatSession.user_id == user_id,
                                ChatSession.created_at >= week_start,
                                ChatSession.created_at < week_end,
                            )
                        )
                    ),
                    ChatMessage.role == "user",
                )
            )
        )
        timestamps = msg_result.scalars().all()
        total_turns = len(timestamps)
        night_turns = sum(1 for ts in timestamps if _is_night_turn(ts))
        night_share = (night_turns / total_turns) if total_turns > 0 else 0.0

        # ----- Get-or-create THIS week's row first -----
        # exclusive_reliance_count / human_support_mention_count for the
        # current week are NOT computed here — they're accumulated
        # incrementally all week via increment_exclusive_reliance() /
        # increment_human_support_mention() (called from the chat route on
        # each turn). This function only needs to READ whatever has already
        # been accumulated so far, never overwrite it. A prior version
        # hardcoded both to 0 here and then unconditionally wrote that 0
        # back onto the row — silently discarding the whole week's
        # incremental counts every time this ran, which meant the
        # exclusive-reliance and human-support-decline signals could never
        # actually fire.
        existing_result = await self.db.execute(
            select(DependencySignal).where(
                and_(
                    DependencySignal.user_id == user_id,
                    DependencySignal.week_start == week_start,
                )
            )
        )
        row = existing_result.scalar_one_or_none()
        if row is None:
            row = DependencySignal(user_id=user_id, week_start=week_start)
            self.db.add(row)

        exclusive_count = row.exclusive_reliance_count or 0

        # ----- Trend signals from PRIOR weeks -----
        trajectory_result = await self.db.execute(
            select(DependencySignal).where(
                and_(
                    DependencySignal.user_id == user_id,
                    DependencySignal.week_start >= week_start - timedelta(weeks=4),
                    DependencySignal.week_start < week_start,
                )
            ).order_by(DependencySignal.week_start.asc())
        )
        prior_signals = trajectory_result.scalars().all()

        # Detect declining human-support mentions over last 2 prior weeks
        human_support_declining = False
        if len(prior_signals) >= _HUMAN_SUPPORT_DECLINE_WEEKS:
            recent = prior_signals[-_HUMAN_SUPPORT_DECLINE_WEEKS:]
            counts = [r.human_support_mention_count for r in recent]
            human_support_declining = all(
                counts[i] > counts[i + 1] for i in range(len(counts) - 1)
            ) and counts[-1] == 0

        level = _composite_level(session_count, night_share, exclusive_count, human_support_declining)

        row.session_count = session_count
        row.total_turn_count = total_turns
        row.night_time_turn_share = round(night_share, 4)
        # exclusive_reliance_count / human_support_mention_count intentionally
        # NOT set here — they're owned by the incrementing helpers above.
        row.signal_level = level
        row.computed_at = datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(row)

        logger.info(
            "DependencySignal computed | user=%s week=%s sessions=%d night=%.0f%% level=%s",
            user_id,
            week_start.date(),
            session_count,
            night_share * 100,
            level,
        )
        return row

    async def increment_exclusive_reliance(self, user_id: str) -> None:
        """Increment exclusive-reliance counter for the current week in place."""
        await self._increment_counter(user_id, "exclusive_reliance_count")

    async def increment_human_support_mention(self, user_id: str) -> None:
        """Increment human-support mention counter for the current week in place."""
        await self._increment_counter(user_id, "human_support_mention_count")

    async def _increment_counter(self, user_id: str, column: str) -> None:
        week_start = _week_start(datetime.now(timezone.utc))
        try:
            result = await self.db.execute(
                select(DependencySignal).where(
                    and_(
                        DependencySignal.user_id == user_id,
                        DependencySignal.week_start == week_start,
                    )
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = DependencySignal(user_id=user_id, week_start=week_start)
                self.db.add(row)
                await self.db.flush()
            current = getattr(row, column, 0) or 0
            setattr(row, column, current + 1)
            await self.db.commit()
        except Exception as e:
            logger.error("_increment_counter(%s, %s) failed: %s", user_id, column, e)

    async def get_signal_level(self, user_id: str) -> str:
        """Return the current week's signal level without recomputing."""
        week_start = _week_start(datetime.now(timezone.utc))
        try:
            result = await self.db.execute(
                select(DependencySignal).where(
                    and_(
                        DependencySignal.user_id == user_id,
                        DependencySignal.week_start == week_start,
                    )
                )
            )
            row = result.scalar_one_or_none()
            return row.signal_level if row else "normal"
        except Exception as e:
            logger.error("get_signal_level failed: %s", e)
            return "normal"

    async def should_prompt_human_support(self, user_id: str) -> bool:
        """True when the current signal warrants a gentle human-support nudge."""
        level = await self.get_signal_level(user_id)
        return level in ("moderate", "high")

    async def get_support_nudge(self, detected_language: Optional[str] = "en") -> str:
        """
        Return a brief, non-alarming nudge toward human support.
        Woven into the therapeutic response at the TherapyNode level.
        """
        if detected_language in ("hi", "hinglish"):
            return (
                "Mujhe aapki bahut parwah hai — aur main chahta hoon ki "
                "aapke paas aur bhi log hon jo samjhein. "
                "Kya aapke kareeb koi hai jisse aap baat kar sakte hain?"
            )
        return (
            "I care a lot about you — and I want to make sure you have "
            "other people in your life too. "
            "Is there someone close to you you could talk to about this?"
        )
