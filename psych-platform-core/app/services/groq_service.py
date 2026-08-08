import json
import logging
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.services.llm_interface import LLMService
from app.services.therapeutic_prompt import build_legacy_prompt as build_therapeutic_system_prompt

logger = logging.getLogger(__name__)


class GroqService(LLMService):
    """
    Fast LLM service via Groq (llama-3.3-70b-versatile).
    ~150ms latency vs ~800ms for Gemini. Used for sentiment node only —
    keeps the hot routing decision off the slower Gemini path.
    """

    def __init__(self, api_key: str | None = None):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=api_key or settings.GROQ_API_KEY,
            temperature=0.0,
            max_retries=2,
        )

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        prompt = (
            f"Analyze this user message: '{text}'. "
            "Return ONLY a JSON object (no markdown, no extra text) with exactly: "
            "'mood' (one of: anxious, depressed, lonely, angry, stressed, fearful, "
            "hopeless, guilty, confused, somatic, tension, anhedonic, calm, neutral) and "
            "'risk_score' (integer 0-10, 10 = immediate suicide risk) and "
            "'cognitive_distortion' (boolean: true if absolute self-judgement, "
            "mind-reading, or catastrophising is present)."
        )
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            if "```" in content:
                parts = content.split("```")
                content = parts[1] if len(parts) > 1 else parts[0]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except Exception as e:
            logger.error("Groq sentiment failed: %s", e)
            return {"mood": "neutral", "risk_score": 0, "cognitive_distortion": False}

    async def analyze_sentiment_raw(self, prompt: str) -> Dict[str, Any]:
        """Accept a fully-formed prompt from SentimentNode and return parsed JSON.

        Deliberately uses the main model (self.llm), NOT a smaller/faster
        model. A prior version used a dedicated fast model here on the theory
        that risk classification is "a much simpler task than response
        generation" — verified only against explicit statements (e.g. "I want
        to end my life"). tests/test_safety.py's indirect-cue probes (the
        bridges test, method-seeking questions) proved that wrong: the fast
        model scored the bridges probe risk_score=0/"calm" outright missing
        it, while this model correctly scored it 8 (crisis threshold). Risk
        scoring on indirect/ambiguous cues needs the same reasoning capacity
        as generation, not less — this is not a place to trade accuracy for
        latency.

        API errors (including 429) are NOT caught here — FallbackLLMService
        handles retries and circuit-breaking."""
        response = await self.llm.ainvoke(prompt)
        content = response.content.strip()
        if "```" in content:
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())

    async def generate_response_for_move(
        self,
        history: List[Dict],
        system_prompt: str,
    ) -> str:
        """Generate a therapeutic response for a specific move + language system prompt."""
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

    async def stream_response_for_move(self, history: List[Dict], system_prompt: str):
        """Streaming sibling of generate_response_for_move — same move-aware,
        exemplar-aware prompt, but yields chunks so a voice/video consumer can
        start speaking on the first sentence instead of waiting for the whole
        response."""
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

    async def is_utterance_complete(self, text: str) -> bool:
        """Semantic half of the voice pipeline's hybrid turn-detector (see
        voice.py) — used alongside Deepgram's acoustic speech_final signal to
        decide whether to respond before the full ~1000ms+ UtteranceEnd
        silence timeout. Deliberately uses the main model, not sentiment_llm:
        empirically, the smaller model was biased toward "COMPLETE" even on
        obviously trailing fragments ("and then I", "well I guess") — 7/14 on
        a hand-built test set, vs 14/14 for this model at near-identical
        latency (~250-450ms) for this task. Fails closed (False) on any
        error, since the caller always has the acoustic UtteranceEnd as a
        fallback — this can only make turn-taking faster, never worse."""
        prompt = (
            'A person is speaking out loud to a voice assistant. Here is what '
            f'they\'ve said so far, transcribed: "{text}"\n'
            'Does this sound like a COMPLETE thought (they are likely finished '
            'speaking for now and expecting a response), or INCOMPLETE (they '
            'sound like they are mid-sentence, trailing off, listing things, '
            'or likely to keep talking)?\n'
            'Respond with exactly one word: COMPLETE or INCOMPLETE.'
        )
        try:
            response = await self.llm.ainvoke(prompt)
            answer = response.content.strip().upper()
            return "INCOMPLETE" not in answer and "COMPLETE" in answer
        except Exception as e:
            logger.error("Turn-completion check failed: %s", e)
            return False

    def _build_messages(
        self, history: List[Dict], context: str, mood: str, language: str = "en"
    ):
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

    async def generate_therapeutic_response(
        self, history: List[Dict], context: str, mood: str
    ) -> str:
        """Full therapeutic response via Groq (fallback when Gemini unavailable)."""
        response = await self.llm.ainvoke(self._build_messages(history, context, mood))
        return response.content

    async def stream_therapeutic_response(
        self, history: List[Dict], context: str, mood: str, language: str = "en"
    ):
        """
        Async generator yielding per-token chunks.
        Groq streams tokens genuinely fast (~100ms to first token),
        making it ideal for real-time audio/video pipelines.
        """
        async for chunk in self.llm.astream(self._build_messages(history, context, mood, language)):
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
