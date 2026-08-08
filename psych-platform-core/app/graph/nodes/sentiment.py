import re
import logging
from app.domain.state import PsychologicalState
from app.services.llm_interface import LLMService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast heuristic language detection (no network call)
# ---------------------------------------------------------------------------

# Devanagari Unicode block: \u0900-\u097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common Hinglish / colloquial Hindi words written in Latin script.
# A hit on 2+ of these → classify as "hinglish".
_HINGLISH_MARKERS = {
    "yaar", "bhai", "behen", "didi", "arre", "accha", "theek", "nahi",
    "haan", "kya", "hai", "tha", "thi", "mujhe", "mera", "meri", "mein",
    "bahut", "acha", "teri", "tera", "tere", "karo", "kar", "abhi",
    "kyun", "kyunki", "lekin", "phir", "toh", "aur", "woh", "yeh",
    "matlab", "samjha", "samjhi", "samajh", "lagta", "lagti", "laga",
    "takleef", "dard", "tension", "pareshan", "udaas", "thaka", "thaki",
    "neend", "khana", "ghar", "dil", "mann", "sar", "pagal", "bc", "bkl",
    "sala", "saala", "chup", "bol", "sun", "dekh", "pata", "nahi pata",
    "kuch", "sab", "sirf", "rehne", "rehna", "aana", "jana", "sona",
    "zindagi", "duniya", "insaan", "log", "ghar", "kaam", "paisa",
}


def _detect_language(text: str) -> str:
    """
    Fast heuristic classification: "hi" | "hinglish" | "en"

    hi        — contains Devanagari script characters
    hinglish  — Latin-script text with >=2 common Hindi/Hinglish markers
    en        — everything else
    """
    if _DEVANAGARI_RE.search(text):
        return "hi"

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    hits = words & _HINGLISH_MARKERS
    if len(hits) >= 2:
        return "hinglish"

    return "en"


# ---------------------------------------------------------------------------
# Cognitive distortion keyword signals (fast pre-check before LLM)
# ---------------------------------------------------------------------------

_DISTORTION_SIGNALS = [
    r"\bhamesha\b",          # "hamesha bura hota hai" (always)
    r"\bkabhi nahi\b",       # "kabhi nahi hoga" (never)
    r"\bsab log\b",          # "sab log mujhe chhod dete hain"
    r"\bkoi nahi\b",         # "koi nahi hai mera"
    r"\bmain hamesha\b",
    r"\bpagal hoon\b",
    r"\bbekaar hoon\b",
    r"\bkisi kaam ka nahi\b",
    r"\bi always\b",
    r"\bi never\b",
    r"\beveryone\s+hates?\b",
    r"\bno\s+one\s+(ever\s+)?cares?\b",
    r"\bi('m|\s+am)\s+(completely\s+)?worthless\b",
    r"\bi('m|\s+am)\s+(a\s+)?failure\b",
    r"\bnothing\s+(ever\s+)?goes\s+right\b",
    r"\bi('m|\s+am)\s+(so\s+)?stupid\b",
    r"\beveryone\s+(always|is)\b",
]

_DISTORTION_RE = re.compile("|".join(_DISTORTION_SIGNALS), re.IGNORECASE)


def _has_distortion_signal(text: str) -> bool:
    return bool(_DISTORTION_RE.search(text))


# ---------------------------------------------------------------------------
# Multimodal hint builder (unchanged from original)
# ---------------------------------------------------------------------------

def _build_multimodal_hint(state: PsychologicalState) -> str:
    parts = []
    audio = state.get("audio_features")
    video = state.get("video_features")

    if audio:
        if audio.get("tone"):
            parts.append(f"voice tone: {audio['tone']}")
        if audio.get("speech_rate"):
            parts.append(f"speech rate: {audio['speech_rate']}")
        if audio.get("energy") is not None:
            energy_label = "low" if audio["energy"] < 0.35 else ("high" if audio["energy"] > 0.7 else "normal")
            parts.append(f"vocal energy: {energy_label} ({audio['energy']:.2f})")

    if video:
        if video.get("dominant_emotion"):
            parts.append(f"facial emotion: {video['dominant_emotion']}")
        if video.get("gaze_avoidance"):
            parts.append("gaze avoidance: yes")
        if video.get("action_units"):
            parts.append(f"facial action units: {', '.join(video['action_units'])}")

    if not parts:
        return ""
    return " | Non-verbal cues — " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# SentimentNode
# ---------------------------------------------------------------------------

# Extended valid moods — includes Indian idiom moods
_VALID_MOODS = {
    "anxious", "depressed", "lonely", "angry", "stressed", "fearful",
    "hopeless", "guilty", "confused", "traumatized", "grieving", "calm",
    "neutral",
    # Indian idiom moods
    "somatic",    # bodily distress (sar bhaari, kamzori, etc.)
    "tension",    # central South Asian distress idiom
    "anhedonic",  # mann nahi lagta presentation
}

_SENTIMENT_PROMPT_TEMPLATE = """\
Analyze the following user message for mood and psychological risk.

User message: '{text}'

Return a JSON object with EXACTLY these keys:
  "mood"       — string, one of: {mood_list}
                 IMPORTANT: If the text contains somatic complaints ('sar bhaari', 'kamzori',
                 'neend nahi', bodily weakness, pain without injury) → use "somatic".
                 If the text uses 'tension' as an emotional descriptor (not physical tension)
                 → use "tension". Do NOT default to "confused" or "neutral" for these.
  "risk_score" — integer 0-10 (0=none, 10=immediate suicide risk).
                 Passive hopelessness with NO ideation content at all — no wish
                 to die, no wish to not exist, just general fed-up-with-life
                 sentiment ('tang aa gaya zindagi se' alone) → score 6-7.
                 PASSIVE SUICIDAL IDEATION — wishing to be dead, wishing to not
                 exist, or wondering what it would be like to disappear/not be
                 there, WITHOUT an explicit desire/intent/plan to act — is its
                 own category, clinically distinct from and more serious than
                 plain hopelessness (this is what PHQ-9's "thoughts you would
                 be better off dead" item screens for). Score 7-8. Examples:
                 'main sochta hoon ki kya hota agar main hota hi nahi' (what
                 would happen if I didn't exist), 'kaash main na hota', 'I
                 wonder what it would be like to just not be here', 'I wish I
                 could just not wake up' — none of these state a desire/intent
                 to act, but all are genuine ideation about non-existence, not
                 mere fatigue-with-life. Do not flatten these to 6 alongside
                 sentiment that has no ideation content at all.
                 Active ideation — ANY expressed desire or intent to end life or
                 die, a plan, or a method — → score 8-10, REGARDLESS of language
                 or how the desire is phrased. This includes Hindi/Hinglish
                 intent phrasing that is easy to under-score if read as mere
                 hopelessness: 'khatam karna chahta hoon' / 'khud ko khatam
                 karna hai' / 'marna chahta hoon' / 'zindagi khatam karna hai'
                 all express DESIRE to end life or die, exactly equivalent to
                 English "I want to end my life" (score 10) or "I want to die"
                 (score 9-10) — score them the same, do not discount them to
                 6-7 just because they are phrased indirectly or wrapped in
                 hopelessness language ('bahut thak gaya hoon... khatam karna
                 chahta hoon sab kuch' is active ideation at 8-10, not passive
                 hopelessness at 6-7, because of the explicit 'khatam karna
                 chahta hoon' desire-to-end clause).
  "cognitive_distortion" — boolean. True if the message contains absolute self-judgement
                 (always/never, worthless, failure), mind-reading ("everyone hates me"),
                 or catastrophising. False otherwise.

Return ONLY valid JSON, no markdown, no explanation.
"""


class SentimentNode:
    def __init__(self, llm_service: LLMService, fallback_llm_service: LLMService | None = None):
        self.llm = llm_service
        self.fallback = fallback_llm_service

    async def __call__(self, state: PsychologicalState) -> dict:
        raw = state["messages"][-1]
        last_message = raw["content"] if isinstance(raw, dict) else getattr(raw, "content", "")
        multimodal_hint = _build_multimodal_hint(state)
        text_with_hints = last_message + multimodal_hint

        # Fast heuristic language detection (no LLM call)
        detected_language = _detect_language(last_message)

        # Fast heuristic distortion pre-check
        heuristic_distortion = _has_distortion_signal(last_message)

        mood_list = ", ".join(sorted(_VALID_MOODS))
        prompt = _SENTIMENT_PROMPT_TEMPLATE.format(
            text=text_with_hints.replace("'", "\\'"),
            mood_list=mood_list,
        )

        analysis = None
        for svc in filter(None, [self.llm, self.fallback]):
            try:
                analysis = await svc.analyze_sentiment_raw(prompt)
                break
            except Exception as e:
                logger.warning("Sentiment failed on %s, trying next: %s", type(svc).__name__, e)

        if analysis is None:
            logger.error("All sentiment services failed — using heuristic defaults")
            return {
                "current_mood": "neutral",
                "risk_score": 0,
                "is_crisis": False,
                "detected_language": detected_language,
                "cognitive_distortion_detected": heuristic_distortion,
            }

        mood = analysis.get("mood", "neutral")
        if mood not in _VALID_MOODS:
            mood = "neutral"

        risk_score = int(analysis.get("risk_score", 0))
        cognitive_distortion = bool(
            analysis.get("cognitive_distortion", heuristic_distortion)
        ) or heuristic_distortion

        if multimodal_hint:
            logger.info(
                "Sentiment (multimodal) | mood=%s risk=%s lang=%s distortion=%s | user=%s",
                mood, risk_score, detected_language, cognitive_distortion, state["user_id"],
            )
        else:
            logger.info(
                "Sentiment | mood=%s risk=%s lang=%s distortion=%s | user=%s",
                mood, risk_score, detected_language, cognitive_distortion, state["user_id"],
            )

        return {
            "current_mood": mood,
            "risk_score": risk_score,
            "is_crisis": risk_score >= 8,
            "detected_language": detected_language,
            "cognitive_distortion_detected": cognitive_distortion,
        }
