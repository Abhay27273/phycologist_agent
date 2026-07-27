# MindBridge: Safety-First AI Mental Health Support

## One-Line Pitch

MindBridge is an AI mental-health support engine that combines real-time conversation, emotional state tracking, clinical knowledge retrieval, and deterministic crisis routing to help people feel heard earlier, safer, and with more continuity than a generic chatbot.

## The Problem

Millions of people need support before they are ready, able, or financially positioned to see a therapist. Existing chatbots can be warm in the moment, but they often fail at the things that matter most in mental-health contexts:

- They forget emotional history between sessions.
- They respond generically instead of grounding replies in evidence-based techniques.
- They handle risk inconsistently when users express self-harm or crisis signals.
- They are difficult to adapt across text, voice, and avatar-based experiences.

MindBridge is built around the idea that mental-health AI should not just answer. It should remember, assess risk, retrieve relevant therapeutic context, and route safely.

## The Solution

MindBridge is a backend "therapeutic brain" for chat, voice, and video clients. It uses a stateful agent workflow to analyze each message, detect mood and risk, retrieve relevant clinical context, and generate a supportive response. When risk is high, it bypasses open-ended generation and uses a deterministic crisis protocol.

The result is not a replacement for therapy. It is an always-available support layer for reflection, coping skills, mood continuity, and safe escalation.

## Why It Is Different

- **Stateful emotional continuity:** LangGraph tracks the conversation state, session summaries, risk level, and longitudinal context across turns.
- **Retrieval-grounded support:** RAG pulls from psychology and counseling resources so responses can be anchored in techniques like CBT, DBT, grounding, emotional regulation, and safety planning.
- **Safety-first routing:** High-risk messages go to a hard-coded crisis node instead of relying on a generative model to improvise.
- **Multimodal-ready:** The API already supports optional audio and video signals such as tone, speech rate, dominant facial emotion, and gaze avoidance.
- **Real-time UX:** Streaming endpoints, sentence-level events, and WebSocket chat make the system suitable for voice agents and avatar companions.
- **Production-aware design:** The deployment plan covers WebSocket-compatible hosting, Postgres-backed graph state, Redis caching/rate limits, and scalable vector search.

## OpenAI Hackathon Angle

OpenAI can turn MindBridge from a strong backend into a deeply natural support experience:

- **Conversational reasoning:** Use OpenAI models for empathetic response generation, structured mood/risk extraction, and summarization.
- **Realtime voice support:** Pair the existing WebSocket architecture with OpenAI realtime speech capabilities for low-latency spoken check-ins.
- **Embeddings and RAG:** Use OpenAI embeddings to index clinical resources and retrieve context for grounded responses.
- **Structured outputs:** Return validated mood, risk score, themes, and recommended coping technique in predictable JSON.
- **Tool calling:** Trigger safety workflows, session summaries, journaling prompts, or escalation resources based on risk and user preference.

## Core Architecture

1. **FastAPI Gateway**
   Receives chat, streaming, and WebSocket requests.

2. **LangGraph Orchestrator**
   Routes each turn through sentiment analysis, risk assessment, therapeutic response generation, summarization, or crisis protocol.

3. **Sentiment and Risk Node**
   Extracts mood, themes, and risk score from the latest message plus optional multimodal signals.

4. **RAG Service**
   Retrieves relevant clinical context from Pinecone or Qdrant using local embeddings and reranking.

5. **Therapy Node**
   Generates a supportive, context-aware response using the conversation state, retrieved knowledge, and prior-session memory.

6. **Crisis Node**
   Uses a deterministic safety message for high-risk cases to avoid hallucinated crisis guidance.

7. **Memory Layer**
   Stores users, chat sessions, messages, summaries, and risk level using Postgres or SQLite for local demo mode.

## Demo Flow

1. A user starts a session and says: "I have been anxious all week and I cannot sleep."
2. MindBridge detects anxiety, assigns a moderate risk score, and retrieves grounding or CBT context.
3. The assistant responds with validation plus a concrete coping step.
4. The user returns later and says: "It is happening again."
5. MindBridge uses the prior session summary to understand what "it" refers to.
6. The user sends a high-risk message.
7. The system immediately routes to the crisis protocol instead of generating an open-ended therapeutic reply.

## Impact

MindBridge targets the gap between "I am struggling" and "I have professional care." It can help users:

- Reflect on emotions in the moment.
- Practice evidence-informed coping strategies.
- Maintain continuity across sessions.
- Receive safer responses during crisis-like language.
- Transition from text to voice or avatar support without rebuilding the intelligence layer.

For providers, schools, wellness platforms, and digital-health teams, MindBridge can become an embeddable support engine that is safer and more stateful than a generic LLM wrapper.

## Responsible AI Positioning

MindBridge should be presented as supportive wellness technology, not a licensed therapist, medical device, or emergency service. The product should:

- Clearly disclose that it is AI.
- Encourage professional care when appropriate.
- Route high-risk content to crisis resources.
- Protect sensitive conversation data with encryption, retention limits, and strict access controls.
- Require legal/compliance review before handling real user health data.

## What We Built

- FastAPI backend with authenticated chat endpoints.
- LangGraph workflow for stateful therapeutic reasoning.
- Mood and risk detection.
- Deterministic crisis intervention node.
- RAG service with Pinecone/Qdrant backend support.
- Streaming chat endpoint.
- Sentence-level streaming for TTS/avatar clients.
- WebSocket chat for real-time clients.
- Session summaries and longitudinal context.
- Deployment blueprint for Azure, AWS, and low-cost demo hosting.

## Ask

We are looking for hackathon support to integrate OpenAI as the primary intelligence layer, polish the real-time voice demo, and validate a safety-first mental-health support workflow that is useful, responsible, and technically ready to scale.

## Closing

MindBridge is built on a simple belief: AI mental-health support should be warm enough to help, structured enough to remember, and cautious enough to know when not to improvise.
