import json
import logging
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.services.llm_interface import LLMService

logger = logging.getLogger(__name__)


class GroqService(LLMService):
    """
    Fast LLM service via Groq (llama-3.3-70b-versatile).
    ~150ms latency vs ~800ms for Gemini. Used for sentiment node only —
    keeps the hot routing decision off the slower Gemini path.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.0,
            max_retries=2,
        )

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        prompt = (
            f"Analyze this user message: '{text}'. "
            "Return ONLY a JSON object (no markdown, no extra text) with exactly: "
            "'mood' (one of: anxious, depressed, lonely, angry, stressed, fearful, "
            "hopeless, guilty, confused, calm, neutral) and "
            "'risk_score' (integer 0-10, 10 = immediate suicide risk)."
        )
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            # Strip any accidental markdown fences
            if "```" in content:
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else parts[0]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except Exception as e:
            logger.error("Groq sentiment failed: %s", e)
            return {"mood": "neutral", "risk_score": 0}

    def _build_messages(self, history: List[Dict], context: str, mood: str):
        messages = [
            SystemMessage(content=(
                f"You are a compassionate, empathetic psychologist AI. "
                f"The user is currently feeling {mood}. "
                f"Use this clinical context as primary evidence: {context}. "
                "Only provide coping suggestions supported by the context. "
                "Keep responses warm, short (under 3 sentences), and human-like."
            ))
        ]
        for msg in history:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            elif hasattr(msg, "type"):
                messages.append(msg)
        return messages

    async def generate_therapeutic_response(
        self, history: List[Dict], context: str, mood: str
    ) -> str:
        """Full therapeutic response via Groq (fallback when Gemini unavailable)."""
        response = await self.llm.ainvoke(self._build_messages(history, context, mood))
        return response.content

    async def stream_therapeutic_response(
        self, history: List[Dict], context: str, mood: str
    ):
        """
        Async generator yielding per-token chunks.
        Groq streams tokens genuinely fast (~100ms to first token),
        making it ideal for real-time audio/video pipelines.
        """
        async for chunk in self.llm.astream(self._build_messages(history, context, mood)):
            if chunk.content:
                yield chunk.content

    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        """Condense this session into 2-3 sentences for long-term memory."""
        lines = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                label = "Client" if m["role"] == "user" else "Therapist"
                lines.append(f"{label}: {m['content']}")
        transcript = "\n".join(lines[-20:])  # cap to last 20 messages to stay within context
        prompt = (
            "Summarize this therapy session in 2-3 sentences. "
            f"The client's current mood is {mood}. "
            "Focus on: key emotional themes, any coping strategies discussed, and overall progress.\n\n"
            f"Session transcript:\n{transcript}\n\nSummary:"
        )
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("Groq summarize_conversation failed: %s", e)
            return ""
