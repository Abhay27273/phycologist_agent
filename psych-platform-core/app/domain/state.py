from typing import TypedDict, List, Annotated, Optional
from operator import add
from pydantic import BaseModel, Field

# --- Pydantic Models for Input/Output Validation ---

class AudioFeatures(BaseModel):
    """Optional non-verbal audio signals from the client (voice agent)."""
    tone: Optional[str] = Field(None, description="e.g. trembling, flat, shaky, steady")
    speech_rate: Optional[str] = Field(None, description="fast | slow | normal")
    energy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalized vocal energy 0-1")

class VideoFeatures(BaseModel):
    """Optional non-verbal video signals from the client (avatar / video call)."""
    dominant_emotion: Optional[str] = Field(None, description="e.g. sad, fearful, neutral")
    gaze_avoidance: Optional[bool] = Field(None, description="True if gaze is consistently averted")
    action_units: Optional[List[str]] = Field(None, description="FACS action units, e.g. ['AU4','AU15']")

class ChatInput(BaseModel):
    user_id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)
    # Phase 3 — optional multimodal signals
    audio_features: Optional[AudioFeatures] = None
    video_features: Optional[VideoFeatures] = None

class ChatOutput(BaseModel):
    response: str
    detected_mood: Optional[str] = None
    risk_level: str

# --- LangGraph State Definitions ---

class StateMessage(TypedDict):
    role: str
    content: str

class PsychologicalState(TypedDict):
    """
    The shared state passed between graph nodes.
    'messages' uses the 'add' reducer to append history rather than overwrite.
    """
    messages: Annotated[List[StateMessage], add]

    # Context
    user_id: str
    session_id: str

    # Analysis
    current_mood: Optional[str]
    risk_score: int  # 0-10 (10 = Immediate Crisis)
    is_crisis: bool

    # Language detection — set by SentimentNode on every turn
    # "en" | "hi" | "hinglish" — drives which prompt register TherapyNode uses
    detected_language: Optional[str]

    # Cognitive distortion flag — set by SentimentNode; triggers reality_test move
    cognitive_distortion_detected: Optional[bool]

    # StrategyNode output — which therapeutic move to execute this turn
    selected_move: Optional[str]
    # Append-only list trimmed to last 3 to prevent move repetition
    last_three_moves: Optional[List[str]]

    # RAG Data
    relevant_context: Optional[str]

    # Phase 2 — Long-term memory
    # Compressed summary of this session (written by SummaryNode every N turns,
    # persisted to chat_sessions.summary by chat.py).
    session_summary: Optional[str]
    # Summaries from prior sessions injected at conversation start by chat.py.
    longitudinal_context: Optional[str]

    # Phase 3 — Multimodal signals (optional, passed from client)
    audio_features: Optional[dict]   # AudioFeatures serialised to dict
    video_features: Optional[dict]   # VideoFeatures serialised to dict