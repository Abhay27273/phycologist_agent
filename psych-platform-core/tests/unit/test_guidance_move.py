"""
Unit tests for the "guidance" move: answering a direct request for direction
instead of deflecting into another reflection + question.

Pure logic, no network, no LLM.
"""
import random

import pytest

from app.graph.nodes.strategy import (
    StrategyNode,
    _guidance_weight,
    explicit_advice_request,
)
from app.services.therapeutic_prompt import (
    MOVE_SET,
    MOVES_THAT_USE_CLINICAL_KB,
    build_therapeutic_system_prompt,
)

# The final turns of the six baseline conversations, where 3 of 6 replies
# deflected instead of helping.
ASKS = [
    "I don't know what to do. What should I do?",
    "Is there anything I can do to calm down before the interview?",
    "What would you suggest I do?",
    "How do I handle this?",
    "mujhe kya karna chahiye?",
    "Please just tell me what to do.",
    "Any tips for sleeping better?",
    "I need some advice about my manager",
    "kya karu yaar, kuch samajh nahi aa raha",
    "how can I stop worrying so much",
    # Devanagari, as returned by the Hindi speech-to-text
    "मुझे आजकल नींद नहीं आती और office की tension बहुत है मुझे क्या करना चाहिए",
    "मैं बहुत परेशान हूँ, क्या करूँ?",
    "कोई सलाह दो",
]

NOT_ASKS = [
    "I've been exhausted at work lately. I stay late almost every day.",
    "I don't know where to start.",
    "I keep imagining that I'll freeze and forget everything.",
    "I don't want advice, I just need to vent.",
    "I'm not looking for tips, I just want someone to listen",
    "I just need to be heard right now",
    "My heart races whenever I think about it.",
    "yaar aajkal neend nahi aati",
    "",
    "advice nahi chahiye, bas sun lo",
    "मुझे सलाह नहीं चाहिए, बस सुन लो",
    "मुझे आजकल नींद नहीं आती",
]


@pytest.mark.parametrize("text", ASKS)
def test_direct_requests_are_detected(text):
    assert explicit_advice_request(text), text


@pytest.mark.parametrize("text", NOT_ASKS)
def test_statements_and_declined_advice_are_not_requests(text):
    assert not explicit_advice_request(text), text


def _state(text, *, risk=1, turn_msgs=1, last_three=None, mood="anxious"):
    msgs = []
    for i in range(turn_msgs - 1):
        msgs.append({"role": "user", "content": f"earlier message {i}"})
        msgs.append({"role": "assistant", "content": "ok"})
    msgs.append({"role": "user", "content": text})
    return {
        "risk_score": risk,
        "current_mood": mood,
        "cognitive_distortion_detected": False,
        "last_three_moves": last_three or [],
        "messages": msgs,
        "relevant_context": "",
        "user_id": "t",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ASKS)
async def test_strategy_forces_guidance_on_a_direct_request(text):
    node = StrategyNode()
    for _ in range(25):  # random selection elsewhere; this must be deterministic
        out = await node(_state(text))
        assert out["selected_move"] == "guidance"


@pytest.mark.asyncio
async def test_guidance_is_honoured_even_right_after_guidance():
    out = await StrategyNode()(_state("What should I do?", last_three=["guidance", "open_question"]))
    assert out["selected_move"] == "guidance"


@pytest.mark.asyncio
async def test_guidance_never_forced_at_crisis_risk():
    # risk >= 8 never reaches StrategyNode in production (CrisisNode handles it),
    # but guidance must not be selected there even if it somehow did.
    for _ in range(25):
        out = await StrategyNode()(_state("What should I do?", risk=9))
        assert out["selected_move"] != "guidance"


@pytest.mark.asyncio
async def test_declined_advice_is_not_overridden():
    random.seed(1)
    for _ in range(40):
        out = await StrategyNode()(_state("I don't want advice, I just need to vent.", turn_msgs=2))
        assert out["selected_move"] != "guidance"


def test_proactive_guidance_weight_rules():
    assert _guidance_weight(1, 1, []) == 0       # too early: listen first
    assert _guidance_weight(3, 1, []) == 0
    assert _guidance_weight(4, 1, []) > 0        # heard for a few turns
    assert _guidance_weight(6, 1, []) > _guidance_weight(4, 1, [])
    assert _guidance_weight(5, 1, ["open_question", "guidance"]) == 0   # not twice in a row
    assert 0 < _guidance_weight(5, 5, []) < _guidance_weight(5, 1, [])  # gentler at elevated risk
    assert _guidance_weight(5, 8, []) == 0       # never at crisis risk


@pytest.mark.asyncio
async def test_proactive_guidance_actually_appears_after_a_few_turns():
    random.seed(7)
    picks = [
        (await StrategyNode()(_state("it keeps happening", turn_msgs=5, risk=1)))["selected_move"]
        for _ in range(300)
    ]
    share = picks.count("guidance") / len(picks)
    assert 0.10 < share < 0.45, f"proactive guidance share {share:.2f} outside the intended band"
    early = [
        (await StrategyNode()(_state("it keeps happening", turn_msgs=2, risk=1)))["selected_move"]
        for _ in range(200)
    ]
    assert "guidance" not in early


def test_guidance_is_a_known_move_that_uses_grounding():
    assert "guidance" in MOVE_SET
    assert "guidance" in MOVES_THAT_USE_CLINICAL_KB


def test_guidance_prompt_asks_for_acknowledge_options_question():
    p = build_therapeutic_system_prompt(context="", mood="anxious", move="guidance", language="en")
    low = p.lower()
    assert "guidance" in low
    assert "one or two small things they could try" in low
    assert "options" in low and "never orders" in low
    assert "curious question" in low
    # With no retrieved grounding it must NOT fall back to the generic
    # "I have no verified information" disclaimer structure.
    assert "do not open with a disclaimer" in low
    # ...and must not contain the generic no-grounding disclaimer scaffold that
    # the other KB moves use ("SECOND sentence: then honestly note you don't have verified...").
    assert "second sentence" not in low


def test_guidance_prompt_uses_grounding_when_available():
    p = build_therapeutic_system_prompt(
        context="Slow diaphragmatic breathing can reduce acute anxiety.",
        mood="anxious", move="guidance", language="en",
    )
    assert "CLINICAL GROUNDING" in p
    assert "diaphragmatic" in p


def test_hinglish_prompt_is_clean_script():
    p = build_therapeutic_system_prompt(context="", mood="anxious", move="guidance", language="hinglish")
    assert "।" not in p, "the prompt itself must not contain the Devanagari danda"
    assert "SCRIPT:" in p
    import re
    assert not re.search(r"[A-Za-z][ऀ-ॿ]|[ऀ-ॿ][A-Za-z]", p), "mixed-script token in prompt"


def test_register_rules_discourage_templated_openings_and_echo():
    p = build_therapeutic_system_prompt(context="", mood="stressed", move="simple_reflection", language="en")
    assert "Vary how you open" in p
    assert "Do not hand the user's own words back" in p


@pytest.mark.asyncio
async def test_summarise_and_check_waits_until_there_is_a_conversation():
    """Turns 1-3: it only recaps and ends on a closed 'does that sound right?'."""
    random.seed(3)
    for turns in (1, 2, 3):
        picks = {
            (await StrategyNode()(_state("I've been feeling stressed at work", turn_msgs=turns, risk=1)))["selected_move"]
            for _ in range(300)
        }
        assert "summarise_and_check" not in picks, f"turn {turns}"
    later = {
        (await StrategyNode()(_state("it keeps happening", turn_msgs=4, risk=1)))["selected_move"]
        for _ in range(400)
    }
    assert "summarise_and_check" in later, "must stay available once there is a conversation to condense"


@pytest.mark.asyncio
async def test_first_reply_always_invites_the_person_to_go_on():
    """sit_with_it asks nothing; as the very first reply it leaves a fresh disclosure unanswered."""
    random.seed(5)
    first = {
        (await StrategyNode()(_state("I've been exhausted at work lately", turn_msgs=1, risk=1)))["selected_move"]
        for _ in range(400)
    }
    assert "sit_with_it" not in first
    second = {
        (await StrategyNode()(_state("My manager keeps piling on tasks", turn_msgs=2, risk=1, last_three=["open_question"])))["selected_move"]
        for _ in range(400)
    }
    assert "sit_with_it" in second, "presence-only turns remain part of the repertoire after the first reply"


def test_prompts_follow_the_language_the_user_is_speaking_now():
    """Calls switch language mid-conversation; history in one language must not pin the reply to it."""
    en = build_therapeutic_system_prompt(context="", mood="anxious", move="open_question", language="en")
    assert "LANGUAGE: reply in English" in en
    assert "even if" in en and "Hindi or Hinglish" in en
    hi = build_therapeutic_system_prompt(context="", mood="anxious", move="open_question", language="hi")
    assert "chahe pehle ki" in hi
