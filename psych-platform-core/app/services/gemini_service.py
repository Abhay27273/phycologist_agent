import logging
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.services.llm_interface import LLMService
from app.services.therapeutic_prompt import build_legacy_prompt as build_therapeutic_system_prompt

logger = logging.getLogger(__name__)

class GeminiService(LLMService):
    def __init__(self):
        # Full-quality model — used for blocking TherapyNode responses
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.7,
            max_retries=2,
        )

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Uses Gemini to detect mood and risk.
        We force JSON output for reliability.
        """
        prompt = (
            f"Analyze the following user text: '{text}'. "
            "Return a JSON object with exactly these keys: "
            "'mood' (string, e.g., 'anxious', 'calm'), "
            "'risk_score' (integer 0-10, where 10 is immediate suicide risk)."
        )

        try:
            # Gemini supports structured output via '.with_structured_output' 
            # or simply prompting for JSON with the right mode.
            response = await self.llm.ainvoke(prompt)
            
            # Simple parsing (In prod, use PydanticOutputParser for strictness)
            import json
            # Gemini usually returns a clean markdown block ```json ... ```
            content = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            
            return data
        except Exception as e:
            logger.error(f"Gemini Sentiment Analysis Failed: {e}")
            return {"mood": "neutral", "risk_score": 0}

    def _build_messages(
        self, history: List[Dict], context: str, mood: str, language: str = "en"
    ):
        """Shared message builder for both blocking and streaming calls."""
        messages = [
            SystemMessage(
                content=build_therapeutic_system_prompt(
                    context, mood, language=language
                )
            )
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

    async def analyze_sentiment_raw(self, prompt: str) -> dict:
        """Accept a fully-formed prompt from SentimentNode and return parsed JSON."""
        try:
            response = await self.llm.ainvoke(prompt)
            import json
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error("Gemini analyze_sentiment_raw failed: %s", e)
            return {"mood": "neutral", "risk_score": 0, "cognitive_distortion": False}

    async def generate_response_for_move(
        self,
        history,
        system_prompt: str,
    ) -> str:
        """Generate a therapeutic response for a specific move + language system prompt."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            elif hasattr(msg, "type"):
                messages.append(msg)
        response = await self.llm.ainvoke(messages)
        return response.content

    async def stream_response_for_move(self, history, system_prompt: str):
        """Streaming sibling of generate_response_for_move — same move-aware,
        exemplar-aware prompt, but yields chunks so a voice/video consumer can
        start speaking on the first sentence instead of waiting for the whole
        response."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            elif hasattr(msg, "type"):
                messages.append(msg)
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    async def generate_therapeutic_response(
        self,
        history: List[Dict],
        context: str,
        mood: str,
    ) -> str:
        """Blocking full-response generation (used by TherapyNode)."""
        messages = self._build_messages(history, context, mood)
        response = await self.llm.ainvoke(messages)
        return response.content

    async def stream_therapeutic_response(
        self,
        history: List[Dict],
        context: str,
        mood: str,
        language: str = "en",
    ):
        """
        Async generator that yields text chunks as Gemini produces them.
        Consumers (audio/video agents) can act on sentence boundaries
        without waiting for the full response.
        """
        messages = self._build_messages(history, context, mood, language)
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        """Condense this session into 2-3 sentences for long-term memory."""
        lines = []
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                label = "Client" if m["role"] == "user" else "Therapist"
                lines.append(f"{label}: {m['content']}")
        transcript = "\n".join(lines[-20:])
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
            logger.error("Gemini summarize_conversation failed: %s", e)
            return ""
