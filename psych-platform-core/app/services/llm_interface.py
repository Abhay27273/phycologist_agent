from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMService(ABC):
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Returns mood and risk score from plain text (legacy)."""
        pass

    async def analyze_sentiment_raw(self, prompt: str) -> Dict[str, Any]:
        """
        Accepts a fully-formed prompt string (built by SentimentNode) and
        returns the parsed JSON dict. Default implementation re-uses
        analyze_sentiment; subclasses may override for efficiency.
        """
        return await self.analyze_sentiment(prompt)

    @abstractmethod
    async def generate_therapeutic_response(
        self,
        history: List[Dict],
        context: str,
        mood: str,
    ) -> str:
        """Generates an empathetic response (legacy — uses default move/language)."""
        pass

    async def generate_response_for_move(
        self,
        history: List[Dict],
        system_prompt: str,
    ) -> str:
        """
        Generate a response given a fully-constructed system prompt.
        TherapyNode calls this after building the prompt via
        build_therapeutic_system_prompt(context, mood, move, language).
        Default implementation calls generate_therapeutic_response with
        empty context/mood so subclasses don't need to override immediately.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_response_for_move"
        )

    @abstractmethod
    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        """Condense a conversation into 2-3 sentences for long-term memory storage."""
        pass
