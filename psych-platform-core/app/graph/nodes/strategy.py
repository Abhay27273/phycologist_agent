"""
StrategyNode — selects the therapeutic move for each turn.

Sits between SentimentNode and TherapyNode. Its output (selected_move) drives
both the style-exemplar retrieval and the generation instruction in TherapyNode.

Move selection inputs:
  - current_mood
  - risk_score (0-7 band; 8+ never reaches here — CrisisNode handles those)
  - cognitive_distortion_detected (→ reality_test move fires)
  - last_three_moves (no consecutive repetition)
  - turn_index (derived from message count — opening/mid/closing shape)
  - clinical_context_available (no psychoeducation without grounding)

Design principles (CONVERSATIONAL_CORE_RESEARCH.md §2.2):
  - 'sit_with_it' must appear ~15-20% of turns (real clinician baseline from AnnoMI)
  - 'reality_test' is the primary sycophancy gate — fires when distortion detected
  - 'open_question' caps at roughly every other turn so the interaction isn't an
    interrogation
  - The fixed "reflect → validate → coping → question" formula is intentionally
    broken by the distribution of moves below
"""

from __future__ import annotations

import random
import re
import logging
from app.domain.state import PsychologicalState
from app.services.therapeutic_prompt import MOVE_SET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Move probability tables
# ---------------------------------------------------------------------------
# Format: list of (move, weight) tuples.
# Weights are relative — they don't have to sum to 100.
#
# Three tables by risk band:
#   low_risk (0-3)  : full range available
#   mid_risk (4-7)  : heavier on reflection/normalising; less psychoeducation
#
# 'psychoeducation' is stripped from the final selection if no clinical context
# is available (see _apply_constraints).

_WEIGHTS_LOW_RISK: list[tuple[str, float]] = [
    ("simple_reflection",    10),
    ("complex_reflection",   12),
    ("affirmation",           8),
    ("open_question",        12),
    ("summarise_and_check",   8),
    ("normalise",             8),
    ("psychoeducation",       6),
    ("sit_with_it",          18),   # 18% baseline — AnnoMI clinician average
    ("reality_test",          0),   # only via distortion override
]

_WEIGHTS_MID_RISK: list[tuple[str, float]] = [
    ("simple_reflection",    14),
    ("complex_reflection",   14),
    ("affirmation",          10),
    ("open_question",         8),
    ("summarise_and_check",  10),
    ("normalise",            12),
    ("psychoeducation",       4),
    ("sit_with_it",          20),
    ("reality_test",          0),   # only via distortion override
]


# Moves whose instructions explicitly end the turn WITHOUT anything for the
# user to answer: sit_with_it ("Do not ask a question"), affirmation ("No
# question."), normalise (a two-sentence statement).
_SILENT_MOVES = frozenset({"sit_with_it", "affirmation", "normalise"})


# ---------------------------------------------------------------------------
# Guidance ("what should I do?") — answering a direct request for direction
# ---------------------------------------------------------------------------
# Observed on the web chat: when someone explicitly asked what to do, the
# reply deflected into another reflection+question in 3 of 6 test
# conversations ("Please just tell me what to do." -> "What's one small thing
# you've tried?"). Being asked for direction and answering only with a
# question reads as evasion. A direct request now selects the guidance move;
# otherwise guidance is offered proactively, but only after the person has had
# a few turns of being heard (turn_index >= 4), at most once per three moves.

_ADVICE_REQUEST_RE = re.compile(
    r"\bwhat (?:should|can|could|shall) i do\b"
    r"|\bwhat (?:would|do|can) you (?:suggest|recommend|advise)\b"
    r"|\bwhat to do\b"
    r"|\b(?:any|some|your)\s+(?:advice|tips?|suggestions?)\b"
    r"|\b(?:give|need|want|share)\s+(?:me\s+)?(?:some\s+)?(?:advice|tips?|suggestions?|guidance|direction)\b"
    r"|\bhow (?:do|can|should|could) i (?:handle|deal|cope|stop|manage|calm|fix|overcome|get through|get over|sleep|focus|stay)\b"
    r"|\bhow to (?:handle|deal|cope|stop|manage|calm|fix|overcome|get through|get over|sleep|focus)\b"
    r"|\bis there (?:anything|something) (?:i|we) can do\b"
    r"|\bcan you (?:help|suggest|advise|recommend)\b"
    # Hinglish
    r"|\bkya karu(?:n)?\b|\bkya karna chahiye\b|\bmujhe kya karna\b|\bkya kare\b"
    r"|\bkoi (?:salah|upay|tarika|solution)\b|\bkaise (?:handle|deal|sambhal|nipat)\b"
    # Devanagari - Sarvam speech-to-text returns Hindi in this script. No \b:
    # Python's \w does not match Devanagari vowel signs, so word boundaries
    # misfire inside these words.
    r"|क्या\s+करूँ|क्या\s+करूं|क्या\s+करु|क्या\s+करना\s+(?:चाहिए|होगा|पड़ेगा)|क्या\s+करें"
    r"|कोई\s+(?:सलाह|उपाय|तरीका|हल|समाधान)|(?:सलाह|सुझाव)\s+(?:दो|दीजिए|दीजिये|चाहिए)"
    r"|कैसे\s+(?:संभालूं|संभालूँ|संभालें|निपटूं|निपटूँ|निपटें)",
    re.IGNORECASE,
)

# "I don't want advice, I just need to vent" must be respected, not overridden.
_NO_ADVICE_RE = re.compile(
    r"\b(?:don'?t|do not|dont|not|no|never)\b[^.?!]{0,25}\b(?:advice|tips?|suggestions?|solutions?|fix(?:ing)?)\b"
    r"|\bjust (?:want|need) (?:to )?(?:vent|talk|be heard|listen)\b"
    # Hinglish / Hindi: "advice nahi chahiye", "bas sun lo", "सलाह नहीं चाहिए", "बस सुन लो"
    r"|\b(?:advice|salah|suggestion|tips?|upay)\s+(?:nahi|nahin|mat)\b|\bbas\s+(?:sun|suno|sunna|baat)\b"
    r"|(?:सलाह|सुझाव|उपाय)\s+(?:नहीं|मत)|बस\s+(?:सुन|सुनो|बात)|सिर्फ़?\s+सुन",
    re.IGNORECASE,
)


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "") or ""
        if getattr(m, "type", None) == "human":
            return getattr(m, "content", "") or ""
    return ""


def explicit_advice_request(text: str) -> bool:
    """True when the user is directly asking what to do (and has not said they don't want advice)."""
    t = (text or "").replace("’", "'")
    if not t.strip() or _NO_ADVICE_RE.search(t):
        return False
    return bool(_ADVICE_REQUEST_RE.search(t))


def _guidance_weight(turn_index: int, risk_score: int, last_three: list[str]) -> float:
    """Proactive guidance weight: none early, none right after guidance, gentler at elevated risk."""
    if risk_score >= 8 or turn_index < 4 or "guidance" in last_three:
        return 0.0
    base = 25.0 if turn_index < 6 else 40.0
    return base * (0.4 if risk_score >= 4 else 1.0)


def _get_weights(risk_score: int) -> list[tuple[str, float]]:
    if risk_score <= 3:
        return list(_WEIGHTS_LOW_RISK)
    return list(_WEIGHTS_MID_RISK)


# ---------------------------------------------------------------------------
# Constraint application
# ---------------------------------------------------------------------------

def _apply_constraints(
    weights: list[tuple[str, float]],
    last_three: list[str],
    cognitive_distortion: bool,
    has_clinical_context: bool,
    turn_index: int,
) -> list[tuple[str, float]]:
    """
    Modify weights in-place (copy) based on situational constraints:
    1. Zero out moves repeated in the last two turns.
    2. Boost/mandate reality_test when distortion detected.
    3. Zero out psychoeducation when no clinical context.
    4. Suppress open_question when it appeared in the last turn
       (interrogation prevention).
    5. Boost sit_with_it on opening turns (turn 0-2) — land before probing.
    """
    adjusted: list[tuple[str, float]] = []

    last_two = set(last_three[-2:]) if len(last_three) >= 2 else set(last_three)
    last_one = last_three[-1] if last_three else None

    for move, weight in weights:
        w = weight

        # 1 — no consecutive same move
        if move in last_two and w > 0:
            w = 0.0

        # 2 — distortion detected: mandate reality_test (very high weight)
        if move == "reality_test" and cognitive_distortion:
            w = 50.0

        # 3 — no psychoeducation without grounding
        if move == "psychoeducation" and not has_clinical_context:
            w = 0.0

        # 4 — suppress open_question if last turn was also open_question
        if move == "open_question" and last_one == "open_question":
            w = 0.0

        # 5 — opening turns (0-2): boost sit_with_it, suppress psychoeducation
        # and bare open_question (a fresh disclosure met with zero acknowledgment
        # and an immediate probe reads as interrogation, not care — even though
        # open_question's "no reflection before it" instruction is correct for
        # later turns where rapport already exists).
        if turn_index <= 2:
            # open_question stays suppressed here — a fresh disclosure met
            # with a bare probe and no acknowledgement reads as interrogation.
            # But sit_with_it is deliberately NOT boosted any more: combined
            # with affirmation/normalise it made opening turns ~55% likely to
            # end with nothing to answer, which over voice means the caller's
            # first real disclosure is met with silence and they don't know
            # whether to continue. Observed live 2026-08-08 (opening turn:
            # "You're feeling really overwhelmed at work right now." — no
            # invitation to go on). At base weight sit_with_it still appears
            # early; the difference is that simple_reflection and
            # complex_reflection — which DO end a substantive disclosure with
            # one focused question — are now the likelier opening, which is
            # what "acknowledge, then invite" actually calls for.
            if move == "psychoeducation":
                w = 0.0
            if move == "open_question":
                w = 0.0
            # affirmation and normalise are premature this early and are also
            # silent, which is why openings were ~55% likely to end with
            # nothing to answer. You cannot name a specific strength or
            # contextualise an experience you have not actually heard yet —
            # "you're doing great" or "a lot of people feel that way" as the
            # reply to someone's first disclosure is exactly the generic
            # reassurance the register rules forbid. Dropping them early
            # leaves sit_with_it as the one silent opening option and makes
            # a reflection-plus-question the likely response instead.
            if move in ("affirmation", "normalise"):
                w = 0.0

        # 5b — the very FIRST reply to a disclosure always invites the person to
        # go on. sit_with_it ("Do not ask a question") as an opening left 2 of 6
        # test conversations with nothing to answer ("It's sitting heavy on you
        # right now."). It stays available from the second turn on, once there
        # is rapport and the ~18-20% clinician baseline applies.
        if turn_index <= 1 and move == "sit_with_it":
            w = 0.0

        # 5c — summarise_and_check needs a conversation to condense. On turns
        # 1-3 it just recaps what was said and ends on a closed "does that sound
        # right?" (measured 2026-11-25: every turn-3 occurrence was a recap with
        # up to 78% of the user's own words reused, in 3 of 6 conversations).
        # From turn 4 there is enough to synthesise.
        if turn_index <= 3 and move == "summarise_and_check":
            w = 0.0

        # 6 — never two turns in a row that give the user nothing to answer.
        # These moves are individually correct and clinically grounded
        # (sit_with_it is deliberately ~18-20% per the AnnoMI baseline), but
        # their combined weight is ~40-45%, so back-to-back silent turns were
        # common. In text that reads as restraint; over VOICE it is dead air —
        # the user cannot tell whether the agent finished, is thinking, or is
        # waiting on them, so the conversation simply stalls. Observed live
        # 2026-08-08: several consecutive voice turns that only restated what
        # the user said ("you're waiting for someone you care about to submit
        # something") with nothing to reply to. Rule 1 already blocks
        # repeating the SAME move; this blocks rotating between different
        # silent ones. sit_with_it keeps its full weight whenever the previous
        # turn did invite a reply, so the distribution is preserved rather
        # than suppressed.
        if move in _SILENT_MOVES and last_one in _SILENT_MOVES:
            w = 0.0

        adjusted.append((move, w))

    return adjusted


def _weighted_choice(weights: list[tuple[str, float]]) -> str:
    moves = [m for m, w in weights if w > 0]
    probs = [w for _, w in weights if w > 0]
    if not moves:
        return "simple_reflection"  # safe fallback
    return random.choices(moves, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# StrategyNode
# ---------------------------------------------------------------------------

class StrategyNode:
    async def __call__(self, state: PsychologicalState) -> dict:
        risk_score = state.get("risk_score", 0)
        mood = state.get("current_mood", "neutral")
        cognitive_distortion = state.get("cognitive_distortion_detected", False)
        last_three: list[str] = list(state.get("last_three_moves") or [])
        messages = state.get("messages", [])
        # Approximate turn index = number of user messages so far
        turn_index = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
        # Clinical context available if relevant_context was set last turn
        has_clinical_context = bool(state.get("relevant_context", "").strip())

        # A direct "what should I do?" is answered, not deflected (crisis,
        # risk >= 8, never reaches this node but is excluded here as well).
        asked_for_advice = risk_score < 8 and explicit_advice_request(_last_user_text(messages))

        if asked_for_advice:
            move = "guidance"
        else:
            weights = _get_weights(risk_score)
            weights = _apply_constraints(
                weights,
                last_three,
                bool(cognitive_distortion),
                has_clinical_context,
                turn_index,
            )
            g = _guidance_weight(turn_index, risk_score, last_three)
            if g > 0:
                weights.append(("guidance", g))

            move = _weighted_choice(weights)

        # Trim last_three to 3 entries
        updated_last_three = (last_three + [move])[-3:]

        logger.info(
            "StrategyNode | move=%s mood=%s risk=%d distortion=%s turn=%d asked_advice=%s | user=%s",
            move, mood, risk_score, cognitive_distortion, turn_index, asked_for_advice,
            state.get("user_id"),
        )

        return {
            "selected_move": move,
            "last_three_moves": updated_last_three,
        }
