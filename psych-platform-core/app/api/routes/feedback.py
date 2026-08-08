"""
Therapeutic alliance and session quality feedback — Phase 5.1.

Instruments (§5.1):
  WAI-SR  — Working Alliance Inventory Short Revised
            12 items, three subscales (goals / tasks / bond), 1–5 scale, α ≈ .92
            Validated in Wysa chatbot study (frontiersin.org/articles/10.3389/fdgth.2022.847991)
  SRS     — Session Rating Scale
            4 items (relationship / goals+topics / approach / overall), 0–10 VAS
            Low friction — submit after every session; 4 sliders in the UI

Automated eval:
  WAI-O-S with LLM as observer-rater (3-round averaging) is supported via
  the /feedback/auto-rate endpoint — for CI regression gating.
  Single-pass LLM judging is too noisy; averaging is the important part.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.database import get_db
from app.infrastructure.models import SessionRating
from app.core.security import decode_token
from jose import JWTError

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_user_id(token: str = Query(...)) -> str:
    try:
        return decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


# ---------------------------------------------------------------------------
# SRS submission
# ---------------------------------------------------------------------------

class SRSInput(BaseModel):
    session_id: str
    relationship: float = Field(..., ge=0, le=10, description="How well did I feel heard, understood, and respected? (0-10)")
    goals_topics: float = Field(..., ge=0, le=10, description="Did we work on what I wanted to work on? (0-10)")
    approach: float = Field(..., ge=0, le=10, description="Did the approach feel right for me? (0-10)")
    overall: float = Field(..., ge=0, le=10, description="Overall, this session was good for me (0-10)")


class SRSResponse(BaseModel):
    rating_id: int
    session_id: str
    srs_overall: float
    message: str


@router.post("/feedback/srs", response_model=SRSResponse, summary="Submit Session Rating Scale")
async def submit_srs(
    body: SRSInput,
    user_id: str = Depends(_get_user_id),
    db: AsyncSession = Depends(get_db),
):
    """4-item Session Rating Scale — submit after every session."""
    rating = SessionRating(
        user_id=user_id,
        session_id=body.session_id,
        instrument="srs",
        srs_relationship=body.relationship,
        srs_goals_topics=body.goals_topics,
        srs_approach=body.approach,
        srs_overall=body.overall,
        submitted_at=datetime.now(timezone.utc),
    )
    db.add(rating)
    await db.commit()
    await db.refresh(rating)
    logger.info(
        "SRS submitted | user=%s session=%s overall=%.1f",
        user_id, body.session_id, body.overall,
    )
    return SRSResponse(
        rating_id=rating.id,
        session_id=body.session_id,
        srs_overall=body.overall,
        message="Thank you — your feedback helps improve the quality of our sessions.",
    )


# ---------------------------------------------------------------------------
# WAI-SR submission
# ---------------------------------------------------------------------------

class WAISRInput(BaseModel):
    session_id: str
    # 12 items, 1-5 each
    # Items 1-4: Goals subscale
    # Items 5-8: Tasks subscale
    # Items 9-12: Bond subscale
    items: list[float] = Field(
        ...,
        min_length=12,
        max_length=12,
        description="12 WAI-SR items, each 1.0-5.0",
    )

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: list[float]) -> list[float]:
        for score in v:
            if not (1.0 <= score <= 5.0):
                raise ValueError(f"WAI-SR item scores must be between 1.0 and 5.0, got {score}")
        return v


class WAISRResponse(BaseModel):
    rating_id: int
    session_id: str
    goals_subscale: float
    tasks_subscale: float
    bond_subscale: float
    total_mean: float


@router.post("/feedback/wai-sr", response_model=WAISRResponse, summary="Submit WAI-SR")
async def submit_wai_sr(
    body: WAISRInput,
    user_id: str = Depends(_get_user_id),
    db: AsyncSession = Depends(get_db),
):
    """12-item Working Alliance Inventory — Short Revised."""
    items = body.items
    goals = statistics.mean(items[0:4])
    tasks = statistics.mean(items[4:8])
    bond = statistics.mean(items[8:12])

    rating = SessionRating(
        user_id=user_id,
        session_id=body.session_id,
        instrument="wai_sr",
        wai_sr_items=items,
        wai_sr_goals=goals,
        wai_sr_tasks=tasks,
        wai_sr_bond=bond,
        submitted_at=datetime.now(timezone.utc),
    )
    db.add(rating)
    await db.commit()
    await db.refresh(rating)
    logger.info(
        "WAI-SR submitted | user=%s session=%s goals=%.2f tasks=%.2f bond=%.2f",
        user_id, body.session_id, goals, tasks, bond,
    )
    return WAISRResponse(
        rating_id=rating.id,
        session_id=body.session_id,
        goals_subscale=round(goals, 2),
        tasks_subscale=round(tasks, 2),
        bond_subscale=round(bond, 2),
        total_mean=round(statistics.mean(items), 2),
    )


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

class AllianceSummary(BaseModel):
    user_id: str
    srs_sessions_rated: int
    srs_mean_overall: Optional[float]
    wai_sr_sessions_rated: int
    wai_sr_mean_bond: Optional[float]
    wai_sr_mean_total: Optional[float]


@router.get("/feedback/summary", response_model=AllianceSummary, summary="Alliance score summary")
async def get_alliance_summary(
    user_id: str = Depends(_get_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregated WAI-SR and SRS scores for the authenticated user."""
    result = await db.execute(
        select(SessionRating).where(SessionRating.user_id == user_id)
    )
    ratings = result.scalars().all()

    srs_ratings = [r for r in ratings if r.instrument == "srs" and r.srs_overall is not None]
    wai_ratings = [r for r in ratings if r.instrument == "wai_sr" and r.wai_sr_items]

    srs_mean = round(statistics.mean(r.srs_overall for r in srs_ratings), 2) if srs_ratings else None
    wai_bond = round(statistics.mean(r.wai_sr_bond for r in wai_ratings if r.wai_sr_bond), 2) if wai_ratings else None
    wai_total = (
        round(statistics.mean(
            statistics.mean(r.wai_sr_items) for r in wai_ratings if r.wai_sr_items
        ), 2)
        if wai_ratings else None
    )

    return AllianceSummary(
        user_id=user_id,
        srs_sessions_rated=len(srs_ratings),
        srs_mean_overall=srs_mean,
        wai_sr_sessions_rated=len(wai_ratings),
        wai_sr_mean_bond=wai_bond,
        wai_sr_mean_total=wai_total,
    )
