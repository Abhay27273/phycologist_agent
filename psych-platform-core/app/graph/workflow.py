import logging
from langgraph.graph import StateGraph, END

from app.domain.state import PsychologicalState
from app.graph.nodes.sentiment import SentimentNode
from app.graph.nodes.strategy import StrategyNode
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

# LLM routing — waterfall with circuit-breaker (FallbackLLMService):
#   Generation: OpenRouter (Nemotron :free, 50/day) → Gemini 2.5 Flash (500/day) → Groq (100k tok/day)
#   Sentiment:  Groq → OpenRouter → Gemini
# Each provider is skipped for 65s automatically after a 429 — no wasted retries.
#
# Sentiment gets its OWN instance (own circuit-breaker state) with Groq
# first, not the shared generation order. Groq's own ~150ms latency (vs
# ~800ms+ for Gemini, and OpenRouter can be considerably slower still) is
# exactly why GroqService.analyze_sentiment_raw docstring calls it the
# service meant to keep "the hot routing decision off the slower Gemini
# path" — but with a single shared OpenRouter-first instance, every
# sentiment call paid the full cost of exhausting OpenRouter+Gemini's
# timeouts before ever reaching Groq. Measured live: this pushed sentiment
# alone to 10-40s whenever OpenRouter/Gemini were rate-limited (see voice
# latency test run 2026-08-05), directly blocking the "meta"/"speaking_started"
# events on the voice path. Generation keeps OpenRouter-first since response
# *quality* (not raw speed) matters more there and Groq's llama-3.3-70b is
# the intentional last-resort/lower-quality option for that role.
# TEMPORARY (2026-08-05): OpenRouter (50/day) and Gemini free tier (actually
# 20 requests/day per model per Google's own 429 response body — the "500/day"
# figure above was wrong) are BOTH fully exhausted for today, confirmed via
# their own rate-limit response headers/bodies (OpenRouter: remaining=0 of 50;
# Gemini: "GenerateRequestsPerDayPerProjectPerModel-FreeTier... quotaValue: 20").
# Neither recovers until its own daily reset. Generation temporarily goes
# Groq-first too so calls don't waste ~1-2s hitting two guaranteed-dead
# providers before reaching the one that actually works. Revert to
# OpenRouter-first once OpenRouter/Gemini quotas reset — Groq's llama-3.3-70b
# is the intentional lower-quality/last-resort option for response text,
# not meant to be primary long-term.
_GROQ_FIRST_FOR_GENERATION_TEMP = True


def _build_groq_pool():
    """Single GroqService, or a RoundRobinLLMService across every configured
    Groq key. Multiple keys directly relieve the actual bottleneck found
    2026-08-06: a single Groq key's ~12K-tokens/minute cap is shared across
    every sentiment AND generation call from both text-chat and voice —
    N keys round-robining multiplies that shared per-minute budget by N."""
    from app.services.groq_service import GroqService
    keys = [
        k for k in (
            settings.GROQ_API_KEY,
            settings.GROQ_API_KEY_2,
            settings.GROQ_API_KEY_3,
            settings.GROQ_API_KEY_4,
        ) if k
    ]
    if not keys:
        return None
    if len(keys) == 1:
        return GroqService(api_key=keys[0])
    from app.services.round_robin_llm_service import RoundRobinLLMService
    logger.info("Groq load-balanced across %d keys (round-robin)", len(keys))
    return RoundRobinLLMService(*(GroqService(api_key=k) for k in keys))


if settings.OPENROUTER_API_KEY:
    from app.services.openrouter_service import OpenRouterService
    from app.services.fallback_llm_service import FallbackLLMService
    _openrouter = OpenRouterService()
    _groq = _build_groq_pool()
    if _groq and _GROQ_FIRST_FOR_GENERATION_TEMP:
        therapy_llm_service = FallbackLLMService(_groq, _openrouter, gemini_service)
    else:
        therapy_llm_service = FallbackLLMService(_openrouter, gemini_service, _groq)
    sentiment_service = (
        FallbackLLMService(_groq, _openrouter, gemini_service) if _groq else therapy_llm_service
    )
    logger.info(
        "LLM waterfall | generation: %s | sentiment: Groq → OpenRouter → Gemini (circuit-breaker on 429)",
        "Groq → OpenRouter → Gemini (TEMP, see 2026-08-05 comment)"
        if (_groq and _GROQ_FIRST_FOR_GENERATION_TEMP)
        else f"OpenRouter ({settings.OPENROUTER_MODEL}) → Gemini → Groq",
    )
elif settings.GROQ_API_KEY:
    from app.services.fallback_llm_service import FallbackLLMService
    # Groq is already first for both roles here, so one shared instance
    # (and one shared circuit-breaker) is enough — no need for the
    # separate-chain split the OPENROUTER_API_KEY branch above needs.
    _primary = FallbackLLMService(_build_groq_pool(), gemini_service)
    therapy_llm_service = _primary
    sentiment_service = _primary
    logger.info("LLM waterfall → Groq → Gemini")
else:
    sentiment_service = gemini_service
    therapy_llm_service = gemini_service
    logger.info("SentimentNode + TherapyNode → Gemini")


# Module-level graph var — None until init_graph() runs in server lifespan.
# chat.py imports this; reads None if lifespan hasn't fired (e.g., tests).
psych_graph = None
_checkpointer_cleanup = None   # callable that closes DB pool / file handle


def _build_workflow():
    """Compile the StateGraph without attaching a checkpointer."""
    sentiment_node = SentimentNode(sentiment_service, fallback_llm_service=gemini_service)
    strategy_node = StrategyNode()
    therapy_node = TherapyNode(therapy_llm_service, rag_service)
    summary_node = SummaryNode(therapy_llm_service)

    workflow = StateGraph(PsychologicalState)
    workflow.add_node("sentiment_analysis", sentiment_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("crisis_protocol", crisis_intervention_node)
    workflow.add_node("therapeutic_response", therapy_node)
    workflow.add_node("summarize", summary_node)

    workflow.set_entry_point("sentiment_analysis")

    def route_based_on_risk(state: PsychologicalState):
        if state.get("is_crisis"):
            return "crisis_protocol"
        return "strategy"

    workflow.add_conditional_edges(
        "sentiment_analysis",
        route_based_on_risk,
        {
            "crisis_protocol": "crisis_protocol",
            "strategy": "strategy",
        },
    )

    # StrategyNode always flows to TherapyNode
    workflow.add_edge("strategy", "therapeutic_response")

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
