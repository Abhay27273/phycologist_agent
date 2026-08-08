# app/infrastructure/models.py
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.infrastructure.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sessions = relationship("ChatSession", back_populates="user")
    mood_trajectory = relationship("MoodTrajectory", back_populates="user")
    user_facts = relationship("UserFact", back_populates="user")
    session_ratings = relationship("SessionRating", back_populates="user")
    dependency_signals = relationship("DependencySignal", back_populates="user")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    risk_level = Column(String, default="LOW")
    summary = Column(Text, nullable=True)
    # Language detected for this session: "en" | "hi" | "hinglish"
    session_language = Column(String, default="en")

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session")
    mood_trajectory = relationship("MoodTrajectory", back_populates="session")
    session_ratings = relationship("SessionRating", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)

    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    detected_mood = Column(String, nullable=True)
    # Therapeutic move used for this assistant turn (null for user messages)
    selected_move = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")


# ---------------------------------------------------------------------------
# Phase 4 — Memory architecture
# ---------------------------------------------------------------------------

class MoodTrajectory(Base):
    """
    Numeric mood time-series per user/session turn.
    Enables trajectory-slope detection for dependency monitoring and
    recurrence-pattern recall (§3.2 / §5.4).
    One row per turn, not per session — granularity matters for slope detection.
    """
    __tablename__ = "mood_trajectory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    mood = Column(String, nullable=False)
    risk_score = Column(Integer, nullable=False, default=0)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="mood_trajectory")
    session = relationship("ChatSession", back_populates="mood_trajectory")


class UserFact(Base):
    """
    Extracted facts about a user — people, relationships, events.
    Each fact has a validity_start / validity_end to handle "broke up with A"
    vs "back together with A" without contradiction (Mem0 / Graphiti pattern).
    Never overwrite; invalidate with validity_end.
    """
    __tablename__ = "user_facts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    # The extracted fact as a natural-language string
    fact_text = Column(Text, nullable=False)
    # Entity mentioned (person name, event label, etc.) — for recall gating
    entity_label = Column(String, nullable=True)
    # Category: person | relationship | event | commitment | preference
    category = Column(String, default="general")

    # Temporal validity (ISO datetime strings; null = open-ended)
    validity_start = Column(DateTime(timezone=True), server_default=func.now())
    validity_end = Column(DateTime(timezone=True), nullable=True)

    # Source session that originated this fact
    source_session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=True)
    # Confidence 0.0-1.0 — from extraction model
    confidence = Column(Float, default=1.0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="user_facts")


# ---------------------------------------------------------------------------
# Phase 5.1 — WAI-SR / Session Rating Scale
# ---------------------------------------------------------------------------

class SessionRating(Base):
    """
    In-app therapeutic alliance and session quality ratings.
    WAI-SR (12-item): goals / tasks / bond subscales, 1-5.
    SRS (4-item):     relationship / goals+topics / approach / overall, 0-10.
    """
    __tablename__ = "session_ratings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    instrument = Column(String, nullable=False)  # "wai_sr" | "srs"

    # SRS fields (4 items, 0-10 each)
    srs_relationship = Column(Float, nullable=True)
    srs_goals_topics = Column(Float, nullable=True)
    srs_approach = Column(Float, nullable=True)
    srs_overall = Column(Float, nullable=True)

    # WAI-SR fields (12 items, 1-5 each) stored as JSON array for flexibility
    wai_sr_items = Column(JSON, nullable=True)  # list of 12 floats
    wai_sr_goals = Column(Float, nullable=True)   # subscale mean
    wai_sr_tasks = Column(Float, nullable=True)
    wai_sr_bond = Column(Float, nullable=True)

    submitted_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="session_ratings")
    session = relationship("ChatSession", back_populates="session_ratings")


# ---------------------------------------------------------------------------
# Phase 5.2 — Dependency monitoring
# ---------------------------------------------------------------------------

class DependencySignal(Base):
    """
    Weekly dependency monitoring signals per user.
    The trajectory-level nature of attachment risk means no single turn looks
    wrong — only the trend does. This table stores the trend. (§5.4)
    """
    __tablename__ = "dependency_signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    week_start = Column(DateTime(timezone=True), nullable=False)  # ISO Monday of the week

    # Session frequency
    session_count = Column(Integer, default=0)
    total_turn_count = Column(Integer, default=0)

    # Night-time concentration (00:00-05:00 share of all turns)
    night_time_turn_share = Column(Float, default=0.0)

    # Exclusive-reliance phrases detected (count this week)
    exclusive_reliance_count = Column(Integer, default=0)

    # Human-support mentions (decline is the alarm)
    human_support_mention_count = Column(Integer, default=0)

    # Composite signal level: "normal" | "moderate" | "high"
    signal_level = Column(String, default="normal")

    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="dependency_signals")
