from app.services.therapeutic_prompt import (
    build_legacy_prompt,
    build_therapeutic_system_prompt,
)


def test_prompt_requires_concrete_exploration_for_substantive_disclosure():
    prompt = build_therapeutic_system_prompt(
        context="CBT grounding and social support planning can help with distress.",
        mood="lonely",
    )

    assert "FIRST-DISCLOSURE STANDARD" in prompt
    assert "do more than paraphrase it" in prompt
    assert "one focused question" in prompt
    assert "CALIBRATED RESPONSE SHAPES" in prompt
    assert "ENGAGEMENT WITHOUT DEPENDENCY" in prompt
    assert "Never claim exclusive closeness" in prompt
    assert "return-worthy continuity" in prompt
    assert "CLINICAL RESPONSE STANDARD" in prompt
    assert "calmly check safety directly" in prompt
    assert "Never switch an" in prompt
    assert "English user into Hindi/Hinglish" in prompt
    # Default reflection turns are deliberately not RAG-grounded. RAG is
    # reserved for psychoeducation/reality-testing, where factual claims need it.
    assert "CBT grounding and social support planning" not in prompt


def test_psychoeducation_includes_retrieved_clinical_grounding():
    prompt = build_therapeutic_system_prompt(
        context="CBT grounding and social support planning can help with distress.",
        mood="anxious",
        move="psychoeducation",
    )

    assert "CLINICAL GROUNDING" in prompt
    assert "CBT grounding and social support planning" in prompt


def test_legacy_streaming_prompt_keeps_hinglish_register():
    prompt = build_legacy_prompt(context="", mood="anxious", language="hinglish")

    assert "Tum ek caring" in prompt
    assert "HINDI / HINGLISH REGISTER" in prompt


def test_prompt_includes_direct_safety_and_medical_boundaries():
    prompt = build_therapeutic_system_prompt(context="", mood="hopeless")

    assert "calmly check safety directly" in prompt
    assert "chest pain, fainting" in prompt
    assert "do not attribute it to anxiety" in prompt
