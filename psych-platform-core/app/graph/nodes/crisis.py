from app.domain.state import PsychologicalState, StateMessage

async def crisis_intervention_node(state: PsychologicalState) -> dict:
    """
    Hard-coded or specialized safety response for high-risk inputs.
    Strictly avoids generative hallucinations during emergencies.
    """
    safety_message = (
        "I'm concerned about what you're going through. I am an AI, not a human, "
        "and I want to make sure you're safe. Please contact a local emergency service "
        "or a suicide prevention hotline immediately."
    )
    
    return {
        "messages": [{"role": "assistant", "content": safety_message}]
    }