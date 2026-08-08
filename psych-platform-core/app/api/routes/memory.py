"""
Memory inspector API — Phase 4.3.

Gives users visibility into and control over what the system remembers.
Three functions:
  1. GET  /memory          — list all active facts for the authenticated user
  2. DELETE /memory/{id}   — hard-delete a specific fact (DPDP erasure right)
  3. GET  /memory/trajectory — mood/risk time-series for the authenticated user

"A 'what I remember about you' screen with per-item delete does three jobs at
once: it converts unease into trust, it is the DPDP-compliant answer to erasure
rights, and therapeutically it hands the user control over what the system
carries about them." — IMPLEMENTATION_PLAN_CONVERSATIONAL_CORE.md §3.3
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database import get_db
from app.services.memory_service import MemoryService
from app.core.security import decode_token
from fastapi import Query

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth helper (mirrors pattern in routes/chat.py)
# ---------------------------------------------------------------------------

def _get_user_id(token: str = Query(..., description="JWT token")) -> str:
    from jose import JWTError
    try:
        return decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class FactItem(BaseModel):
    id: int
    fact_text: str
    entity_label: Optional[str] = None
    category: str
    confidence: float


class FactListResponse(BaseModel):
    user_id: str
    facts: list[FactItem]
    total: int


class ForgetResponse(BaseModel):
    deleted: bool
    fact_id: int


class TrajectoryPoint(BaseModel):
    mood: str
    risk_score: int
    recorded_at: Optional[str] = None


class TrajectoryResponse(BaseModel):
    user_id: str
    trajectory: list[TrajectoryPoint]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/memory", response_model=FactListResponse, summary="List remembered facts")
async def list_memory(
    user_id: str = Depends(_get_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return all currently valid facts the system holds about the authenticated user."""
    svc = MemoryService(db)
    facts = await svc.get_active_facts(user_id)
    return FactListResponse(
        user_id=user_id,
        facts=[FactItem(**f) for f in facts],
        total=len(facts),
    )


@router.delete("/memory/{fact_id}", response_model=ForgetResponse, summary="Delete a remembered fact")
async def forget_fact(
    fact_id: int,
    user_id: str = Depends(_get_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Hard-delete a specific fact. This is a permanent erasure — the fact cannot
    be recovered. Required for DPDP Act 2023 compliance (§4.5).
    """
    svc = MemoryService(db)
    deleted = await svc.forget(user_id, fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found or access denied")
    return ForgetResponse(deleted=True, fact_id=fact_id)


@router.get("/memory/trajectory", response_model=TrajectoryResponse, summary="Mood trajectory")
async def mood_trajectory(
    user_id: str = Depends(_get_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's mood and risk_score time-series (last 5 sessions)."""
    svc = MemoryService(db)
    trajectory = await svc.get_recent_trajectory(user_id)
    return TrajectoryResponse(
        user_id=user_id,
        trajectory=[TrajectoryPoint(**t) for t in trajectory],
    )
