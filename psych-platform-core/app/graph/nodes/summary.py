import logging
from app.domain.state import PsychologicalState
from app.services.llm_interface import LLMService

logger = logging.getLogger(__name__)

# Trigger summarization every N messages (user + assistant combined).
# At 2 messages/turn this fires after every 5 exchanges.
SUMMARY_INTERVAL = 10


class SummaryNode:
    """
    Compresses the full conversation history into a 2-3 sentence summary stored
    in PsychologicalState.session_summary.  chat.py persists this to
    chat_sessions.summary so the insight survives server restarts and informs
    future sessions via longitudinal_context.

    Triggered by a conditional edge in workflow.py whenever
    len(messages) % SUMMARY_INTERVAL == 0.
    """

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def __call__(self, state: PsychologicalState) -> dict:
        messages = state.get("messages", [])
        mood = state.get("current_mood", "neutral")
        try:
            summary = await self.llm.summarize_conversation(messages, mood)
            if summary:
                logger.info(
                    "Session summarized | session=%s | msgs=%d | chars=%d",
                    state.get("session_id"), len(messages), len(summary),
                )
            return {"session_summary": summary} if summary else {}
        except Exception as e:
            logger.error("SummaryNode failed: %s", e)
            return {}
