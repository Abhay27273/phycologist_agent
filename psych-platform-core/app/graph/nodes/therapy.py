import logging
from app.domain.state import PsychologicalState
from app.services.llm_interface import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

# Moods that reliably need clinical context (skip RAG for neutral/calm).
_CLINICAL_MOODS = {"anxious", "depressed", "lonely", "angry", "stressed",
                   "fearful", "hopeless", "guilty", "confused"}


def _needs_rag(message: str, mood: str) -> bool:
    """
    Return False for short/trivial messages that don't benefit from RAG.
    Saves 1.5-4s of cross-encoder CPU time per request.
    """
    words = message.split()
    if len(words) < 4 and mood not in _CLINICAL_MOODS:
        return False
    return True


class TherapyNode:
    """
    Core therapeutic response node.
    Generates empathetic, evidence-based responses using the LLM.
    """
    def __init__(self, llm_service: LLMService, rag_service: RAGService):
        self.llm = llm_service
        self.rag = rag_service

    async def __call__(self, state: PsychologicalState) -> dict:
        """
        Generates a therapeutic response based on conversation history and mood.
        Returns a partial state update with the assistant's message.
        """
        try:
            # Extract context
            conversation_history = state["messages"]
            current_mood = state.get("current_mood", "neutral")

            # Find the most recent user message
            last_user_message = ""
            for msg in reversed(conversation_history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break
                elif hasattr(msg, "type") and msg.type == "human":
                    last_user_message = getattr(msg, "content", "")
                    break

            # Skip RAG for short/low-signal messages — saves 1.5-4s cross-encoder time.
            if _needs_rag(last_user_message, current_mood):
                clinical_context = await self.rag.retrieve_clinical_context(
                    query=last_user_message,
                    mood=current_mood,
                )
            else:
                clinical_context = ""
                logger.info("RAG skipped (short/trivial message) | mood=%s", current_mood)

            # Merge longitudinal context (past sessions) with clinical RAG context.
            longitudinal_context = state.get("longitudinal_context") or ""
            if longitudinal_context and clinical_context:
                relevant_context = (
                    f"[Previous sessions]\n{longitudinal_context}\n\n"
                    f"[Clinical evidence]\n{clinical_context}"
                )
            elif longitudinal_context:
                relevant_context = f"[Previous sessions]\n{longitudinal_context}"
            else:
                relevant_context = clinical_context

            # Generate therapeutic response
            response = await self.llm.generate_therapeutic_response(
                history=conversation_history,
                mood=current_mood,
                context=relevant_context
            )
            
            logger.info(f"Therapy Response Generated | User: {state['user_id']} | Mood: {current_mood}")
            
            return {
                "messages": [{"role": "assistant", "content": response}],
                "relevant_context": clinical_context,
            }
            
        except Exception as e:
            logger.error(f"Therapy Node Failed: {str(e)}")
            # Fallback response
            return {
                "messages": [{
                    "role": "assistant", 
                    "content": "I'm here to listen. Could you tell me more about what you're experiencing?"
                }]
            }
