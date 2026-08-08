import json
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.core.config import settings
from app.services.llm_interface import LLMService
from app.services.therapeutic_prompt import build_legacy_prompt as build_therapeutic_system_prompt

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


class OpenRouterService(LLMService):
    """
    LLM service via OpenRouter (OpenAI-compatible API).
    Exceptions are NOT swallowed here — callers (FallbackLLMService) handle retries.
    """

    def __init__(self, model: str | None = None):
        _model = model or getattr(settings, "OPENROUTER_MODEL", _DEFAULT_MODEL)
        _common = dict(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            max_retries=1,
            default_headers={
                "HTTP-Referer": "https://psych-platform.local",
                "X-Title": "Psych Platform",
            },
        )
        self.llm = ChatOpenAI(model=_model, temperature=0.7, **_common)
        self._sentiment_llm = ChatOpenAI(model=_model, temperature=0.0, **_common)

    def _parse_json(self, content: str) -> dict:
        content = content.strip()
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())

    def _build_messages(self, history, system_prompt: str) -> list:
        msgs = [SystemMessage(content=system_prompt)]
        for msg in history:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    msgs.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    msgs.append(AIMessage(content=msg["content"]))
            elif hasattr(msg, "type"):
                msgs.append(msg)
        return msgs

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        prompt = (
            f"Analyze this user message: '{text}'. "
            "Return ONLY a JSON object (no markdown, no extra text) with exactly: "
            "'mood' (one of: anxious, depressed, lonely, angry, stressed, fearful, "
            "hopeless, guilty, confused, somatic, tension, anhedonic, calm, neutral) and "
            "'risk_score' (integer 0-10, 10 = immediate suicide risk) and "
            "'cognitive_distortion' (boolean)."
        )
        response = await self._sentiment_llm.ainvoke(prompt)
        return self._parse_json(response.content)

    async def analyze_sentiment_raw(self, prompt: str) -> Dict[str, Any]:
        response = await self._sentiment_llm.ainvoke(prompt)
        return self._parse_json(response.content)

    async def generate_therapeutic_response(
        self, history: List[Dict], context: str, mood: str
    ) -> str:
        system_prompt = build_therapeutic_system_prompt(context=context, mood=mood)
        messages = self._build_messages(history, system_prompt)
        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate_response_for_move(self, history, system_prompt: str) -> str:
        messages = self._build_messages(history, system_prompt)
        response = await self.llm.ainvoke(messages)
        return response.content

    async def stream_response_for_move(self, history, system_prompt: str):
        """Streaming sibling of generate_response_for_move — same move-aware,
        exemplar-aware prompt (built externally by the caller, unlike
        stream_therapeutic_response below which still uses the legacy
        pre-move prompt builder), yielding chunks for low-latency voice/video."""
        messages = self._build_messages(history, system_prompt)
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    async def stream_therapeutic_response(
        self, history, mood: str, context: str, language: str = "en"
    ):
        system_prompt = build_therapeutic_system_prompt(
            context=context, mood=mood, language=language
        )
        messages = self._build_messages(history, system_prompt)
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        history_text = "\n".join(
            f"{m.get('role','?').upper()}: {m.get('content','')}"
            for m in messages if isinstance(m, dict)
        )
        prompt = (
            "Summarize this therapy conversation in 2-3 sentences, "
            f"focusing on key themes and emotional state.\n\n{history_text}"
        )
        response = await self.llm.ainvoke(prompt)
        return response.content
