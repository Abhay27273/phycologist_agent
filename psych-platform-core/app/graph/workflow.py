import logging
from langgraph.graph import StateGraph, END

from app.domain.state import PsychologicalState
from app.graph.nodes.sentiment import SentimentNode
from app.graph.nodes.therapy import TherapyNode
from app.graph.nodes.crisis import crisis_intervention_node
from app.graph.nodes.summary import SummaryNode, SUMMARY_INTERVAL
from app.services.gemini_service import GeminiService
from app.services.rag_service import RAGService
from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Module-level service singletons ---
# Shared by the graph AND the streaming endpoint in chat.py.

gemini_service = GeminiService()
rag_service = RAGService()

# Use Groq for ALL LLM calls when a key is configured:
#   - Groq (llama-3.3-70b) ~150ms sentiment, ~300ms generation
#   - Gemini 2.5 Flash has a thinking phase that buffers all tokens (~8-10s)
#     making it unsuitable for low-latency paths even in "blocking" mode
if settings.GROQ_API_KEY:
    from app.services.groq_service import GroqService
    sentiment_service = GroqService()
    logger.info("SentimentNode + TherapyNode → Groq (fast path enabled)")
else:
    sentiment_service = gemini_service
    logger.info("SentimentNode + TherapyNode → Gemini (set GROQ_API_KEY to enable fast path)")


# Module-level graph var — None until init_graph() runs in server lifespan.
# chat.py imports this; reads None if lifespan hasn't fired (e.g., tests).
psych_graph = None
_checkpointer_cleanup = None   # callable that closes DB pool / file handle


def _build_workflow():
    """Compile the StateGraph without attaching a checkpointer."""
    sentiment_node = SentimentNode(sentiment_service)
    # Use sentiment_service (Groq when available) for therapy too — Gemini's
    # thinking mode adds 8-10s of latency even for simple messages.
    therapy_node = TherapyNode(sentiment_service, rag_service)
    summary_node = SummaryNode(sentiment_service)

    workflow = StateGraph(PsychologicalState)
    workflow.add_node("sentiment_analysis", sentiment_node)
    workflow.add_node("crisis_protocol", crisis_intervention_node)
    workflow.add_node("therapeutic_response", therapy_node)
    workflow.add_node("summarize", summary_node)

    workflow.set_entry_point("sentiment_analysis")

    def route_based_on_risk(state: PsychologicalState):
        if state.get("is_crisis"):
            return "crisis_protocol"
        return "therapeutic_response"

    workflow.add_conditional_edges(
        "sentiment_analysis",
        route_based_on_risk,
        {
            "crisis_protocol": "crisis_protocol",
            "therapeutic_response": "therapeutic_response",
        },
    )

    def route_after_therapy(state: PsychologicalState):
        """Trigger SummaryNode every SUMMARY_INTERVAL messages."""
        msg_count = len(state.get("messages", []))
        if msg_count > 0 and msg_count % SUMMARY_INTERVAL == 0:
            return "summarize"
        return "__end__"

    workflow.add_conditional_edges(
        "therapeutic_response",
        route_after_therapy,
        {"summarize": "summarize", "__end__": END},
    )
    workflow.add_edge("crisis_protocol", END)
    workflow.add_edge("summarize", END)
    return workflow


async def init_graph():
    """
    Called once from the FastAPI lifespan startup hook.

    Picks the right persistent checkpointer based on DATABASE_URL:
      sqlite+aiosqlite://  → AsyncSqliteSaver  (dev, single-process)
      postgresql://        → AsyncPostgresSaver (prod, multi-worker)

    Sets the module-level `psych_graph` and `_checkpointer_cleanup` so
    chat.py can use `psych_graph` without knowing about the checkpointer.
    """
    global psych_graph, _checkpointer_cleanup

    workflow = _build_workflow()
    db_url = settings.DATABASE_URL

    if db_url.startswith("postgresql") or db_url.startswith("postgres://"):
        # Production: persistent PostgreSQL checkpointer
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # psycopg3 uses a plain DSN (no dialect prefix)
        dsn = db_url.replace("postgresql+asyncpg://", "postgresql://").replace(
            "postgresql+psycopg://", "postgresql://"
        )
        pool = AsyncConnectionPool(
            conninfo=dsn,
            max_size=5,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        psych_graph = workflow.compile(checkpointer=checkpointer)

        async def _cleanup():
            await pool.close()

        _checkpointer_cleanup = _cleanup
        logger.info("LangGraph checkpointer → AsyncPostgresSaver (PostgreSQL)")

    else:
        # Development: SQLite-backed checkpointer (single-process only)
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        # Extract the SQLite file path from the URL.
        # sqlite+aiosqlite:///./psych_db.sqlite → ./psych_db.sqlite
        sqlite_path = db_url.split("///", 1)[-1]
        # Store the checkpointer in a different file than the main DB to avoid
        # SQLite write-lock contention with SQLAlchemy's connection pool.
        checkpoint_path = sqlite_path.replace(".sqlite", "_checkpoints.sqlite")

        # from_conn_string() is a context manager; __aenter__ returns the actual saver.
        _saver_ctx = AsyncSqliteSaver.from_conn_string(checkpoint_path)
        checkpointer = await _saver_ctx.__aenter__()
        psych_graph = workflow.compile(checkpointer=checkpointer)

        async def _cleanup():
            await _saver_ctx.__aexit__(None, None, None)

        _checkpointer_cleanup = _cleanup
        logger.info(
            "LangGraph checkpointer → AsyncSqliteSaver (%s)", checkpoint_path
        )


async def close_graph():
    """Called from FastAPI lifespan shutdown to release DB connections."""
    if _checkpointer_cleanup:
        await _checkpointer_cleanup()
