# Psych Platform Core

AI-powered psychological support platform using LangGraph orchestration.

## Architecture

- **API Layer**: FastAPI endpoints with dependency injection
- **Core**: Configuration, logging, and exception handling
- **Domain**: Business entities and state definitions
- **Graph**: LangGraph workflow orchestration with crisis detection
- **Services**: LLM and RAG service abstractions

## Setup

```bash
# Install dependencies
poetry install

# Copy environment variables
cp .env.example .env

# Run the application
poetry run uvicorn app.api.server:app --reload
```

## Testing

```bash
poetry run pytest
```
