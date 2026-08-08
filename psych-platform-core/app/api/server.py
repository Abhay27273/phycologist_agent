import os
import sys
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# --- WINDOWS FIXES (must be before any native library imports) ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Prevents access-violation crash when PyTorch (used by HuggingFace) and
    # LangGraph/Pydantic load conflicting OpenMP/BLAS DLLs in the same process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
# -----------------------------------------------------------------

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.limiter import limiter
from app.api.routes import chat, health, voice, memory, feedback
from app.api.routes import auth as auth_routes
from app.graph.workflow import init_graph, close_graph

async def _warm_up_sentiment_llm() -> None:
    """Fires a throwaway sentiment-analysis call so the first real voice/chat
    request doesn't pay the LLM client's one-time connection-setup cost
    (~800-1000ms, observed empirically for a fresh ChatGroq instance) — runs
    in the background, overlapping with the embedding model still loading."""
    try:
        from app.graph.workflow import sentiment_service
        await sentiment_service.analyze_sentiment("hello")
    except Exception:
        pass


async def _warm_up_smart_turn() -> None:
    """Loads Smart Turn v3's ONNX session + downloads the model file (if not
    already cached) at startup, so the first real voice call doesn't pay
    that cost mid-conversation."""
    try:
        from app.services.turn_detector import smart_turn_detector
        await smart_turn_detector.is_utterance_complete(b"\x00\x00" * 16000)
    except Exception:
        pass


async def _warm_up_style_exemplars() -> None:
    """Pre-populate RAGService's style-exemplar cache for every
    (move, register, valence) combination.

    Measured live 2026-08-07: an uncached lookup cost 6.8-18.2s DURING an
    active voice call — a short BGE-base embedding plus a filtered Qdrant
    scan (run twice via the valence-relaxation loop) is fast in isolation,
    but under a saturated event loop (~50 audio packets/sec inbound plus TTS
    pumping) the GIL contention inflates it enormously. That was the
    dominant voice latency AND the cause of Deepgram closing the STT socket
    with 1011 ("did not receive audio within the timeout window"), since the
    /ws/voice receive loop couldn't forward mic audio for >10s.

    Caching alone would still leave the FIRST turn of each move paying that
    cost mid-call, so the combinations are computed here instead — at
    startup, on an idle loop, where each is only a few hundred ms. The key
    space is tiny and fully enumerable, so this is exhaustive rather than
    predictive."""
    try:
        from app.graph.workflow import rag_service
        from app.services.therapeutic_prompt import MOVE_SET
        # Let the app finish coming up and serve /health before starting —
        # this work is CPU-bound and GIL-hungry, so an immediate exhaustive
        # sweep delays readiness by minutes (observed).
        await asyncio.sleep(5)
        for move in sorted(MOVE_SET):
            for register in ("en", "hinglish-casual"):
                for valence in ("neutral", "low"):
                    for k in (2, 3):  # voice uses k=2, TherapyNode k=3
                        await rag_service.retrieve_style_exemplars(
                            move=move, register=register,
                            affect_valence=valence, k=k,
                        )
                        # Yield between combinations. Without this the sweep
                        # monopolises the loop exactly the way the mid-call
                        # lookups did — a warmup must never itself become the
                        # stall it exists to prevent.
                        await asyncio.sleep(0.2)
    except Exception:
        # Best-effort warmup — a failure here must never block startup, and
        # the cache simply fills lazily on first use instead.
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    print(f"Server starting on {sys.platform}...")

    # Redis pool — optional; RAGService falls back to local dict if unavailable.
    redis_client = None
    if settings.REDIS_URL and not settings.REDIS_URL.startswith("memory://"):
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=False)
            await redis_client.ping()
            print("Redis connected.")
            # Inject into the module-level rag_service singleton
            from app.graph.workflow import rag_service
            rag_service._redis = redis_client
        except Exception as exc:
            print(f"Redis unavailable ({exc}), using in-process RAG cache.")
            redis_client = None

    await init_graph()
    asyncio.create_task(_warm_up_sentiment_llm())
    asyncio.create_task(_warm_up_smart_turn())
    asyncio.create_task(_warm_up_style_exemplars())
    yield
    await close_graph()
    if redis_client:
        await redis_client.aclose()

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        lifespan=lifespan,
        docs_url=f"{settings.API_PREFIX}/docs",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Security: CORS — split comma-separated string into list
    cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files have no Cache-Control header by default, so browsers use
    # their own heuristic caching and can silently keep serving an old JS/CSS
    # file after a normal refresh — force revalidation on every request
    # (still cheap: ETag/Last-Modified above mean an unchanged file gets a
    # 304, not a full re-fetch).
    @app.middleware("http")
    async def no_cache_static(request, call_next):
        response = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response

    # Routes (registered before static mount so API paths take priority)
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth_routes.router, prefix=settings.API_PREFIX, tags=["Auth"])
    app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["Chat"])
    app.include_router(voice.router, prefix=settings.API_PREFIX, tags=["Voice"])
    app.include_router(memory.router, prefix=settings.API_PREFIX, tags=["Memory"])
    app.include_router(feedback.router, prefix=settings.API_PREFIX, tags=["Feedback"])

    # Serve chat UI — mount last so API routes win on any overlap
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app

app = create_application()