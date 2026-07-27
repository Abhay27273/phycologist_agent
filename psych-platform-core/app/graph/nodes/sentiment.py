import logging
from app.domain.state import PsychologicalState
from app.services.llm_interface import LLMService

logger = logging.getLogger(__name__)


def _build_multimodal_hint(state: PsychologicalState) -> str:
    """
    Convert optional audio/video features into a plain-English hint appended
    to the sentiment prompt so the LLM can adjust mood/risk scoring.
    Returns empty string when no features are present.
    """
    parts = []
    audio = state.get("audio_features")
    video = state.get("video_features")

    if audio:
        if audio.get("tone"):
            parts.append(f"voice tone: {audio['tone']}")
        if audio.get("speech_rate"):
            parts.append(f"speech rate: {audio['speech_rate']}")
        if audio.get("energy") is not None:
            energy_label = "low" if audio["energy"] < 0.35 else ("high" if audio["energy"] > 0.7 else "normal")
            parts.append(f"vocal energy: {energy_label} ({audio['energy']:.2f})")

    if video:
        if video.get("dominant_emotion"):
            parts.append(f"facial emotion: {video['dominant_emotion']}")
        if video.get("gaze_avoidance"):
            parts.append("gaze avoidance: yes")
        if video.get("action_units"):
            parts.append(f"facial action units: {', '.join(video['action_units'])}")

    if not parts:
        return ""
    return " | Non-verbal cues — " + "; ".join(parts) + "."


class SentimentNode:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def __call__(self, state: PsychologicalState) -> dict:
        """
        Analyzes the last user message for emotion and risk.
        Incorporates multimodal audio/video hints when available.
        Returns a partial state update.
        """
        last_message = state["messages"][-1]["content"]
        multimodal_hint = _build_multimodal_hint(state)
        text_with_hints = last_message + multimodal_hint

        try:
            analysis = await self.llm.analyze_sentiment(text_with_hints)

            if multimodal_hint:
                logger.info(
                    "Sentiment (multimodal) | mood=%s risk=%s | user=%s",
                    analysis.get("mood"), analysis.get("risk_score"), state["user_id"],
                )
            else:
                logger.info("Sentiment | %s | user=%s", analysis, state["user_id"])

            return {
                "current_mood": analysis.get("mood", "neutral"),
                "risk_score": analysis.get("risk_score", 0),
                "is_crisis": analysis.get("risk_score", 0) >= 8,
            }
        except Exception as e:
            logger.error("Sentiment Analysis Failed: %s", str(e))
            return {"current_mood": "neutral", "risk_score": 0, "is_crisis": False}