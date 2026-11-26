"""
Deterministic last-resort crisis-cue detector.

WHY THIS EXISTS
Crisis routing used to depend entirely on an LLM's risk_score (>= 8 -> crisis
template). That is only as reliable as whichever free-tier model happens to
answer: a model swap (llama-3.3-70b -> gpt-oss-120b) took the indirect-cue
probes in tests/test_safety.py (the "tall buildings / standing on the edge"
case, the "too many sleeping pills" case) from risk 8 to risk 0, and when every
provider fails SentimentNode silently scored 0. Plain regexes cannot be
swapped out, rate-limited or timed out, so they sit IN FRONT of every model.

DESIGN RULES
* High precision over recall. This is a safety NET under the LLM, not a
  replacement for it: anything it misses still goes to the model as before.
  A false positive costs one unnecessary (but harmless) crisis message; a miss
  can cost far more, so ambiguous-but-plausible phrasings lean toward firing.
* It can only RAISE risk, never lower it.
* It never logs the user's text, only rule names.
* English, Hinglish (Latin script) and Devanagari Hindi.

REVIEW NOTE
The phrase lists below are an engineering first pass, not a clinically
validated instrument. They should be reviewed by a clinician before this is
relied on, and extended from real (anonymised) misses.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches the existing routing threshold: risk_score >= 8 -> crisis path.
CRISIS_THRESHOLD = 8
EXPLICIT_FLOOR = 9   # stated ideation / intent / method-seeking
INDIRECT_FLOOR = 8   # indirect cues (e.g. height + "calming") -> at the threshold


@dataclass(frozen=True)
class CrisisCue:
    rule: str
    floor: int


def _normalize(text: str) -> str:
    t = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
    )
    return " ".join(t.lower().split())


# A negator shortly before a match ("I would never kill myself", "I don't want
# to die") means the statement is protective, not ideation. Only applied to
# rules marked negatable; the LLM still sees the message either way.
_NEGATION_BEFORE = re.compile(r"(?:\bnever\b|\bnot\b|n't\b|\bno\b)[^.!?;,]{0,25}$")


def _negated(text: str, start: int) -> bool:
    return bool(_NEGATION_BEFORE.search(text[max(0, start - 40):start]))


_HEIGHT_OBJ = r"(?:bridge|building|roof|rooftop|balcony|cliff|terrace|tower|window|flyover|overpass|ledge)"
_SLEEP_MEDS = (
    r"(?:pills?|tablets?|capsules?|sleeping|paracetamol|acetaminophen|"
    r"painkillers?|insulin|tylenol|antidepressants?)"
)

# (name, floor, negatable, pattern)
_RULES: list[tuple[str, int, bool, str]] = [
    # --- explicit ideation / intent (English) ---------------------------------
    ("self_kill", EXPLICIT_FLOOR, True,
     r"\b(?:kill|killing|hang|hanging|drown|drowning|shoot|shooting|poison|poisoning)"
     r"\s+(?:myself|my\s*self)\b(?!\s+(?:laughing|with\s+laugh))"),
    ("end_life", EXPLICIT_FLOOR, True,
     r"\b(?:end|ending|take|taking)\s+(?:my\s+own\s+life|my\s+life)\b|\bend\s+it\s+all\b"),
    ("want_die", EXPLICIT_FLOOR, True,
     r"\b(?:want|wanna|wish|ready|planning|plan|decided)\s+(?:to\s+|2\s+)?"
     r"(?:die|be\s+dead|not\s+be\s+(?:alive|here)|disappear\s+forever)\b"
     r"|\bwish\s+(?:i|that\s+i)\s+(?:was|were|could\s+be)\s+dead\b"
     r"|\bwish\s+(?:i|that\s+i)\s+(?:could\s+)?(?:just\s+)?(?:die|disappear|not\s+wake\s+up|not\s+exist)\b"),
    ("better_dead", EXPLICIT_FLOOR, True,
     r"\bbetter\s+off\s+(?:dead|without\s+me)\b"),
    ("no_reason_to_live", EXPLICIT_FLOOR, False,
     r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living|go(?:ing)?\s+on(?:\s+living)?|being\s+alive)\b"
     r"|\b(?:don'?t|do\s+not)\s+(?:see|have)\s+(?:a|any)\s+(?:reason|point)\s+(?:to|in)\s+"
     r"(?:live|living|go(?:ing)?\s+on(?:\s+living)?)\b"),
    ("dont_want_to_live", EXPLICIT_FLOOR, False,
     r"\b(?:don'?t|do\s+not|no\s+longer)\s+want\s+to\s+(?:live|be\s+alive|exist)\b"
     r"|\b(?:don'?t|do\s+not)\s+want\s+to\s+wake\s+up(?=\s*(?:$|[.,!?;]))"),
    ("suicidal", EXPLICIT_FLOOR, True,
     r"\bi\s*(?:am|'m|feel|was|have\s+been)\s+(?:so\s+|really\s+|very\s+)?suicidal\b"
     r"|\b(?:feeling|feel)\s+suicidal\b"
     r"|\bsuicidal\s+(?:thoughts?|ideation|urges?|feelings?)\b"
     r"|\b(?:thinking|thoughts?)\s+(?:about|of)\s+(?:suicide|killing\s+myself|ending\s+(?:it|my\s+life))\b"
     r"|\bcommit(?:ting)?\s+suicide\b"),
    ("self_harm", EXPLICIT_FLOOR, True,
     r"\bself[\s-]?harm(?:ing)?\b"
     r"|\b(?:want|going|need|urge|urges|tempted|thinking\s+(?:about|of))\s+(?:to\s+)?"
     r"(?:hurt|harm|cut|burn|injure)\s+(?:myself|my\s*self)\b"),

    # --- means / method seeking (the "sleeping pills, just asking" probe) -----
    ("overdose_query", EXPLICIT_FLOOR, False,
     rf"\b(?:too\s+many|overdos(?:e|ing|ed)|lethal\s+dose|fatal\s+dose|enough\s+to\s+(?:die|kill))\b"
     rf"[^.?!]{{0,50}}\b{_SLEEP_MEDS}\b"
     rf"|\b{_SLEEP_MEDS}\b[^.?!]{{0,50}}\b(?:overdos(?:e|ing)|lethal|fatal|to\s+die|to\s+kill\s+(?:myself|me))\b"),
    ("method_seeking", EXPLICIT_FLOOR, False,
     r"\b(?:painless|easiest|quickest|fastest|best|least\s+painful)\s+(?:way|ways|method|methods)\s+to\s+"
     r"(?:die|kill\s+(?:myself|yourself|oneself)|commit\s+suicide|end\s+(?:it|my\s+life))\b"
     r"|\bways?\s+to\s+(?:die|kill\s+myself|commit\s+suicide)\b"
     r"|\bhow\s+(?:do|can|could|would|to)\s+(?:i\s+|someone\s+|you\s+)?"
     r"(?:kill\s+myself|commit\s+suicide|hang\s+myself|end\s+my\s+life|die\s+painlessly)\b"
     r"|\bhow\s+long\s+(?:does\s+it|would\s+it)\s+take\s+to\s+(?:die|bleed\s+out)\b"),

    # --- heights (the "tall buildings / standing on the edge" probe) ----------
    ("edge_standing", INDIRECT_FLOOR, False,
     r"\bstand(?:ing)?\s+(?:on|at|near)\s+the\s+(?:edge|ledge)\b"
     r"(?!\s+of\s+(?:my\s+|the\s+|a\s+)?(?:seat|chair|bed|sofa|couch|table|desk|pool|stage|bath|tub|mattress|road|sidewalk))"),
    ("jump_from_height", INDIRECT_FLOOR, False,
     rf"\b(?:jump|jumping|leap|leaping|throw\s+myself|throwing\s+myself)\s+(?:off|from|out\s+of)\s+"
     rf"(?:a\s+|an\s+|the\s+|my\s+)?(?:\w+\s+)?{_HEIGHT_OBJ}\b"
     r"|\b(?:jump|jumping|throw\s+myself|throwing\s+myself)\s+(?:in\s+front\s+of|under)\s+"
     r"(?:a\s+|the\s+)?(?:train|metro|truck|bus)\b"),
    ("height_ideation", INDIRECT_FLOOR, False,
     r"\b(?:tall\s+buildings?|high\s+buildings?|rooftop|bridge)\b[^.?!]{0,100}\b(?:edge|ledge|jump|jumping)\b"
     r"|\b(?:jump|jumping)\b[^.?!]{0,40}\b(?:calm(?:ing)?|peace(?:ful)?|relief|tempt(?:ing|ed)?|appeal(?:ing|s)?)\b"),

    # --- Hinglish (Latin script) ----------------------------------------------
    ("hinglish_intent", EXPLICIT_FLOOR, False,
     r"\b(?:zindagi|jeevan|jeena|jeene|thak\s+gay[ai]|tang\s+aa)\b.{0,100}"
     r"\bkhatam\s+kar(?:na|ne|dena|du|dun)?\s*(?:chahta|chahti|chahte|hai|ka\s+(?:man|mann))\b"
     r"|\b(?:zindagi|jeevan)\s+(?:ko\s+)?khatam\b"
     r"|\bkhud\s+ko\s+(?:khatam|maar)\b"
     r"|\bmar(?:na|ne)\s+(?:chahta|chahti|ka\s+(?:man|mann))\b"
     r"|\bmar\s+j(?:a|aa)(?:un|u)\b"
     r"|\bmar\s+j(?:a|aa)na\s+(?:chahta|chahti)\b"
     r"|\baa?tm(?:a)?\s*hatya\b"
     r"|\bsuicide\s+kar(?:na|ne|lu|lunga|lungi)\b"
     r"|\bjeene\s+ka\s+(?:man|mann)\s+nahi\b"
     r"|\bjeena\s+nahi\s+(?:chahta|chahti)\b"),

    # --- Devanagari Hindi -------------------------------------------------------
    ("devanagari_intent", EXPLICIT_FLOOR, False,
     r"आत्महत्या"
     r"|मरना\s+चाहत[ाी]"
     r"|खुद\s+को\s+(?:खत्म|ख़त्म|मार)"
     r"|(?:ज़िंदगी|जिंदगी|ज़िन्दगी|जिन्दगी)\s+(?:को\s+)?(?:खत्म|ख़त्म)"
     r"|मर\s+जाना\s+चाहत[ाी]"
     r"|जीना\s+नहीं\s+चाहत[ाी]"
     r"|जीने\s+का\s+मन\s+नहीं"),
]

_COMPILED = [
    (name, floor, negatable, re.compile(pat, re.IGNORECASE))
    for name, floor, negatable, pat in _RULES
]


def detect_crisis_cues(text: str) -> list[CrisisCue]:
    """Every rule that fires on `text`, strongest first. Empty list if none."""
    if not text or not text.strip():
        return []
    t = _normalize(text)
    hits: list[CrisisCue] = []
    for name, floor, negatable, rx in _COMPILED:
        for m in rx.finditer(t):
            if negatable and _negated(t, m.start()):
                continue
            hits.append(CrisisCue(name, floor))
            break
    hits.sort(key=lambda c: -c.floor)
    return hits


def crisis_risk_floor(text: str) -> int:
    """Minimum risk_score this text warrants on its own (0 = no rule fired)."""
    cues = detect_crisis_cues(text)
    return cues[0].floor if cues else 0


def apply_crisis_floor(text: str, risk_score: int, *, source: str = "") -> int:
    """Return max(risk_score, rule floor). Logs rule names (never the text)."""
    cues = detect_crisis_cues(text)
    if not cues:
        return risk_score
    floor = cues[0].floor
    if floor > risk_score:
        logger.warning(
            "Crisis rules raised risk %s -> %s | rules=%s | source=%s",
            risk_score, floor, ",".join(c.rule for c in cues), source or "?",
        )
    return max(risk_score, floor)
