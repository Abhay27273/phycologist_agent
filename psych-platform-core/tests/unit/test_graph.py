import pytest
from unittest.mock import AsyncMock, MagicMock
from app.graph.nodes.sentiment import SentimentNode
from app.domain.state import PsychologicalState

@pytest.mark.asyncio
async def test_sentiment_node_detects_crisis():
    # Arrange
    mock_llm = MagicMock()
    mock_llm.analyze_sentiment = AsyncMock(return_value={"mood": "despair", "risk_score": 9})
    
    node = SentimentNode(llm_service=mock_llm)
    
    state: PsychologicalState = {
        "messages": [{"role": "user", "content": "I can't take it anymore"}],
        "user_id": "test_user",
        "current_mood": None
    }
    
    # Act
    result = await node(state)
    
    # Assert
    assert result["is_crisis"] is True
    assert result["risk_score"] == 9
    mock_llm.analyze_sentiment.assert_called_once()