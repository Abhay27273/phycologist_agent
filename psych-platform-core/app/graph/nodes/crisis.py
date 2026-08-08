from app.domain.state import PsychologicalState

# Deterministic safety templates — no LLM interpolation, ever.
# Language selected based on detected_language from SentimentNode.

_CRISIS_EN = (
    "I can hear that you're in a very difficult place right now, and I'm genuinely concerned.\n\n"
    "Please reach out to someone who can help immediately:\n\n"
    "Tele-MANAS: 14416 (also 1-800-891-4416)\n"
    "Government of India — free, 24 hours, 7 days a week.\n"
    "Available in Hindi, English, and 20+ Indian languages.\n"
    "Trained counsellors, with psychiatrist escalation when needed.\n\n"
    "If you are in immediate danger: call 112.\n\n"
    "You do not have to go through this alone. I'll be here when you're ready to talk."
)

_CRISIS_HI = (
    "Main samajh sakta hoon ki abhi bahut mushkil waqt chal raha hai. Mujhe tumhari chinta ho rahi hai.\n\n"
    "Please abhi kisi se baat karo jo sach mein madad kar sake:\n\n"
    "Tele-MANAS: 14416 (ya 1-800-891-4416)\n"
    "Bharat Sarkar ki seva — bilkul muft, 24 ghante, saat din.\n"
    "Hindi, English aur 20 se zyada bhaarateey bhashaon mein available.\n"
    "Trained counsellors hain, zaroorat padne par psychiatrist se bhi baat ho sakti hai.\n\n"
    "Agar abhi turant khatara ho: 112 call karo.\n\n"
    "Akele mat raho is waqt. Jab baat karni ho, main yahaan hoon."
)


def crisis_message_for(language: str | None) -> str:
    """Return the deterministic, language-matched crisis response."""
    return _CRISIS_HI if language in ("hi", "hinglish") else _CRISIS_EN


async def crisis_intervention_node(state: PsychologicalState) -> dict:
    """
    Hard-coded safety response for risk_score >= 8.
    Strictly avoids LLM generation — deterministic only.
    Language selected from detected_language; defaults to English.
    """
    message = crisis_message_for(state.get("detected_language"))

    return {
        "messages": [{"role": "assistant", "content": message}]
    }
