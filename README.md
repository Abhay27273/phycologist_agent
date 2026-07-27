# 🧠 AI Psychologist Platform (Core API)

> **A Clinical-Grade, Multimodal Cognitive Architecture for Mental Health Support.**

## 📖 Overview
This project is an advanced AI backend designed to simulate therapeutic interaction. Unlike standard chatbots, it utilizes a **Stateful Cognitive Architecture (LangGraph)** to maintain emotional permanence, track user psychological profiles, and apply clinical frameworks (CBT/DBT) via **Retrieval Augmented Generation (RAG)**.

The system is built to be the central "Brain" that powers:
1.  **Text Clients** (Web/Mobile Chat)
2.  **Voice Agents** (Real-time Speech-to-Speech)
3.  **Video Avatars** (Visual/Emotional Telehealth)

---

## 🏗 High-Level Architecture (HLD)

The system follows a **Modular "Brain" Architecture**. The API serves as the central intelligence node, while Audio and Video layers act as peripheral adapters.

```mermaid
graph TD
    User((User)) -->|HTTPS/WSS| API_Gateway[FastAPI Gateway]
    
    subgraph "The Brain (Core Intelligence)"
        API_Gateway -->|Request| Orchestrator[LangGraph Orchestrator]
        Orchestrator -->|1. Analyze| SentimentNode[Sentiment & Risk Engine]
        Orchestrator -->|2. Reason| TherapyNode[Therapeutic Logic]
        Orchestrator -->|3. Safety| CrisisNode[Crisis Protocol]
    end
    
    subgraph "The Heart (Memory & Knowledge)"
        TherapyNode <-->|Retrieve Context| VectorDB[(Pinecone Vector DB)]
        SentimentNode -->|Read/Write| SQL[(PostgreSQL DB)]
        VectorDB <-->|Ingest| Books[Psychology Textbooks]
    end
    
    subgraph "External Services"
        TherapyNode -->|Inference| LLM[Google Gemini 2.5 Flash]
        CrisisNode -->|Alert| Alerts[Emergency Services API]
    end
Core ComponentsThe Orchestrator (LangGraph): Manages the conversation flow. It is not linear; it loops, reflects, and routes based on emotional state.The Memory (PostgreSQL): Stores accurate, long-term session history, user profiles, and risk assessments.The Wisdom (Pinecone RAG): Stores vector embeddings of clinical literature (Freud, CBT Manuals), allowing the AI to "cite" professional sources.🛠 Tech StackComponentTechnologyReasoningLanguagePython 3.12+Standard for AI/ML Engineering.API FrameworkFastAPIAsync/Await support for high-concurrency (Audio/Video).Cognitive EngineLangGraphStateful, cyclical agent flows (essential for therapy).LLM ProviderGoogle Gemini 2.5Low latency, large context window, cost-effective.DatabasePostgreSQL (AsyncPG)Robust relational data integrity for patient records.Vector DBPineconeServerless, scalable knowledge retrieval.MigrationsAlembicVersion control for database schema.🚀 Setup & InstallationPrerequisitesPython 3.10+PostgreSQL (Local or Cloud)Google AI Studio API Key1. Environment ConfigurationCreate a .env file in the root directory:Ini, TOMLAPP_NAME="Psych-Platform-Core"
ENVIRONMENT="local"
DATABASE_URL="postgresql://user:pass@127.0.0.1:5432/psych_db"
GOOGLE_API_KEY="AIzaSy..."
PINECONE_API_KEY="pcsk_..."
PINECONE_INDEX_NAME="psych-brain"
2. InstallationBash# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Database MigrationBash# Initialize DB Tables
alembic upgrade head
4. Run ServerBash# Start API (Auto-reloads on save)
uvicorn app.api.server:app --reload