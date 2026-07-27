Project Structure
Plaintext

psych-platform-core/
├── app/
│   ├── api/            # Routes (Chat, Health, Session)
│   ├── core/           # Config, Logging, Security
│   ├── domain/         # Pydantic Models & State Definitions
│   ├── graph/          # LangGraph Nodes & Workflow Logic
│   ├── infrastructure/ # DB Models & Connection Logic
│   └── services/       # External Integrations (Gemini, Pinecone)
├── data/               # Raw PDF Knowledge Base
├── docs/               # Architecture Documentation (LLD)
├── migrations/         # Alembic SQL Scripts
└── scripts/            # Utilities (RAG Ingestion)

---

### **2. Low-Level Design (LLD) - `docs/LLD.md`**

Create a folder named `docs` and create `LLD.md` inside it. This serves as the blueprint for your code.

```markdown
# 📐 Low-Level Design (LLD) Document

## 1. Data Models (ER Diagram Schema)

The database adheres to **3rd Normal Form** to ensure data integrity. We use **SQLAlchemy ORM** with Async support.

### **Entity: User (`users`)**
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | PK, UUID | Unique User ID (from Auth Provider) |
| `email` | VARCHAR | Unique, Not Null | User contact |
| `created_at` | DATETIME | Default: Now | Registration timestamp |

### **Entity: ChatSession (`chat_sessions`)**
Represents a distinct conversation thread (e.g., "Tuesday Therapy").
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | VARCHAR | PK, UUID | Unique Session ID |
| `user_id` | VARCHAR | FK (`users.id`) | Owner of the session |
| `risk_level` | VARCHAR | Default: 'LOW' | Current safety assessment |
| `summary` | TEXT | Nullable | Long-term memory summary of session |

### **Entity: ChatMessage (`chat_messages`)**
Individual turns in the conversation.
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PK, AutoInc | Sequential ID |
| `session_id` | VARCHAR | FK (`chat_sessions.id`) | Parent session |
| `role` | VARCHAR | 'user' / 'assistant' | Speaker identity |
| `content` | TEXT | Not Null | The actual text/payload |
| `detected_mood` | VARCHAR | Nullable | Metadata from Sentiment Engine |
| `timestamp` | DATETIME | Default: Now | Time of message |

---

## 2. Component Design

### **A. Service Layer (`app/services`)**
To maintain **Separation of Concerns**, business logic does not touch raw HTTP or Database code.

#### `GeminiService` (Class)
* **Responsibility:** Interface with Google's LLM.
* **Methods:**
    * `analyze_sentiment(text: str) -> Dict`: Returns mood and risk score (0-10).
    * `generate_response(history, context) -> str`: Produces the final empathetic text.

#### `RAGService` (Class)
* **Responsibility:** Interface with Pinecone Vector DB.
* **Methods:**
    * `retrieve_clinical_context(query: str, mood: str) -> str`: Fetches relevant book excerpts based on semantic similarity.

---

### **B. The Cognitive Graph (`app/graph`)**
The application logic is defined as a Directed Cyclic Graph (DCG).

#### **State Definition (`PsychologicalState`)**
The shared memory passed between nodes:
```python
class PsychologicalState(TypedDict):
    messages: List[BaseMessage]   # Conversation history
    current_mood: str             # 'Anxious', 'Calm', etc.
    is_crisis: bool               # Safety trigger
    rag_context: str              # Retrieved book knowledge
Sequence Flow
Sentiment Node:

Input: Last User Message.

Action: Calls GeminiService.analyze_sentiment.

Output: Updates current_mood and is_crisis.

Router (Conditional Edge):

IF is_crisis == True -> Go to Crisis Node.

ELSE -> Go to Therapy Node.

Therapy Node:

Action 1: Calls RAGService.retrieve_clinical_context.

Action 2: Calls GeminiService.generate_response (injecting history + RAG context).

Output: Returns final string response.

3. API Contracts (Interface Specs)
POST /api/v1/chat
The primary endpoint for all client interactions.

Request (ChatInput):

JSON

{
  "user_id": "u_12345",
  "session_id": "s_98765",
  "message": "I feel lost today."
}
Response (ChatOutput):

JSON

{
  "response": "I'm sorry to hear that. Can you tell me what's making you feel that way?",
  "detected_mood": "sadness",
  "risk_level": "LOW"
}
4. Security & Safety
Input Sanitization: All Pydantic models strip whitespace and validate max lengths to prevent injection attacks.

Risk Gating: The CrisisNode is a hard-coded safety valve. It does not rely purely on LLM generation; it uses deterministic templates for suicide/harm prevention to ensure reliability.

Data Isolation: User sessions are strictly siloed by user_id.