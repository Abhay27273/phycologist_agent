import re
import logging
from app.domain.state import PsychologicalState
from app.services.llm_interface import LLMService
from app.services.rag_service import RAGService
from app.services.therapeutic_prompt import (
    build_therapeutic_system_prompt,
    MOVES_THAT_USE_CLINICAL_KB,
)

logger = logging.getLogger(__name__)

# Moods that reliably need clinical context (skip RAG for neutral/calm).
_CLINICAL_MOODS = {
    "anxious", "depressed", "lonely", "angry", "stressed",
    "fearful", "hopeless", "guilty", "confused",
    "somatic", "tension", "anhedonic", "traumatized", "grieving",
}


# Moods with no real distress signal — everything else is treated as "low"
# affect valence, matching the convention style exemplars are tagged with
# (see scripts/ingest_annomi.py — AnnoMI has no real valence annotation, so
# exemplars are bucketed low/neutral/high off a crude keyword heuristic).
_NON_DISTRESS_MOODS = {"calm", "neutral"}


def _affect_valence_for_mood(mood: str) -> str:
    return "neutral" if (mood or "").lower() in _NON_DISTRESS_MOODS else "low"


def _register_for_language(language: str) -> str:
    # Only "en" exemplars exist today (AnnoMI is English-only) — Hindi/Hinglish
    # turns will simply get no exemplars back until a Hindi-labeled source
    # exists. Returning a distinct tag rather than falling back to "en" keeps
    # that boundary honest instead of silently mixing register.
    return "en" if (language or "en") == "en" else "hinglish-casual"


def _needs_rag(message: str, mood: str, move: str) -> bool:
    """
    Skip RAG for moves that don't use clinical context, or short/trivial messages.
    Saves 1.5-4s of cross-encoder CPU time per request.
    """
    if move not in MOVES_THAT_USE_CLINICAL_KB:
        return False
    words = message.split()
    if len(words) < 4 and mood not in _CLINICAL_MOODS:
        return False
    return True


# ---------------------------------------------------------------------------
# Anti-echo guard
# ---------------------------------------------------------------------------

def _content_word_overlap(a: str, b: str) -> float:
    """Fraction of content words in `a` that also appear in `b`."""
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "it", "its", "this", "that", "these", "those", "you", "your", "i", "me",
        "my", "we", "our", "they", "their", "he", "she", "his", "her",
        # Hindi stopwords
        "hai", "hain", "tha", "thi", "the", "ko", "ke", "ki", "ka", "se",
        "mein", "par", "aur", "toh", "kya", "yeh", "woh", "jo", "main",
    }
    def content_words(text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-zA-Z\u0900-\u097F]{3,}", text.lower())
            if w not in stop
        }
    a_words = content_words(a)
    b_words = content_words(b)
    if not a_words:
        return 0.0
    return len(a_words & b_words) / len(a_words)


_ECHO_THRESHOLD = 0.35


class TherapyNode:
    def __init__(self, llm_service: LLMService, rag_service: RAGService):
        self.llm = llm_service
        self.rag = rag_service

    async def __call__(self, state: PsychologicalState) -> dict:
        try:
            conversation_history = state["messages"]
            current_mood = state.get("current_mood", "neutral")
            move = state.get("selected_move") or "simple_reflection"
            language = state.get("detected_language") or "en"

            # Find the most recent user message
            last_user_message = ""
            for msg in reversed(conversation_history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break
                elif hasattr(msg, "type") and msg.type == "human":
                    last_user_message = getattr(msg, "content", "")
                    break

            # RAG — only fetch clinical context if this move uses it
            if _needs_rag(last_user_message, current_mood, move):
                clinical_context = await self.rag.retrieve_clinical_context(
                    query=last_user_message,
                    mood=current_mood,
                )
                logger.info("RAG retrieved | move=%s mood=%s", move, current_mood)
            else:
                clinical_context = ""
                logger.info("RAG skipped | move=%s mood=%s", move, current_mood)

            # Merge longitudinal context (past sessions) with clinical context
            longitudinal_context = state.get("longitudinal_context") or ""
            if longitudinal_context and clinical_context:
                merged_context = (
                    f"[Previous sessions]\n{longitudinal_context}\n\n"
                    f"[Clinical evidence]\n{clinical_context}"
                )
            elif longitudinal_context:
                merged_context = f"[Previous sessions]\n{longitudinal_context}"
            else:
                merged_context = clinical_context

            # Retrieve real-clinician style exemplars for this move, filtered
            # by metadata (move + mood valence + register) — never by semantic
            # similarity to the user's message, so no one else's specific
            # situation can leak into this response.
            style_exemplars = await self.rag.retrieve_style_exemplars(
                move=move,
                affect_valence=_affect_valence_for_mood(current_mood),
                register=_register_for_language(language),
                k=2,
            )

            # Build move-specific, language-aware system prompt
            system_prompt = build_therapeutic_system_prompt(
                context=merged_context,
                mood=current_mood,
                move=move,
                language=language,
                style_exemplars=style_exemplars,
            )

            # Generate response
            if hasattr(self.llm, "generate_response_for_move"):
                response = await self.llm.generate_response_for_move(
                    history=conversation_history,
                    system_prompt=system_prompt,
                )
            else:
                # Fallback for services that haven't implemented generate_response_for_move
                response = await self.llm.generate_therapeutic_response(
                    history=conversation_history,
                    mood=current_mood,
                    context=merged_context,
                )

            # Anti-echo guard: if response is too similar to an injected exemplar,
            # regenerate once without the exemplar block.
            # (Style exemplars not yet injected — guard is a no-op but ready for Phase 2.)
            if _content_word_overlap(response, last_user_message) > _ECHO_THRESHOLD:
                logger.warning(
                    "Echo detected (overlap=%.2f) | move=%s — regenerating",
                    _content_word_overlap(response, last_user_message), move,
                )
                response = await self.llm.generate_response_for_move(
                    history=conversation_history,
                    system_prompt=system_prompt + "\n\nIMPORTANT: Do not echo or paraphrase the user's exact words.",
                ) if hasattr(self.llm, "generate_response_for_move") else response

            logger.info(
                "TherapyNode | move=%s lang=%s mood=%s | user=%s",
                move, language, current_mood, state["user_id"],
            )

            return {
                "messages": [{"role": "assistant", "content": response}],
                "relevant_context": clinical_context,
            }

        except Exception as e:
            logger.error("TherapyNode failed: %s", str(e))
            # Language-aware fallback
            language = state.get("detected_language") or "en"
            if language in ("hi", "hinglish"):
                fallback = "Main yahan hoon. Thoda aur batao — kya chal raha hai?"
            else:
                fallback = "I'm here. Could you tell me a little more about what's going on?"
            return {
                "messages": [{"role": "assistant", "content": fallback}]
            }
