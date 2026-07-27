from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMService(ABC):
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Returns mood and risk score."""
        pass

    @abstractmethod
    async def generate_therapeutic_response(
        self,
        history: List[Dict],
        context: str,
        mood: str,
    ) -> str:
        """Generates the empathetic response."""
        pass

    @abstractmethod
    async def summarize_conversation(self, messages: List[Dict], mood: str) -> str:
        """Condense a conversation into 2-3 sentences for long-term memory storage."""
        pass