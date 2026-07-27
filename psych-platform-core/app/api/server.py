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
from app.api.routes import chat, health
from app.api.routes import auth as auth_routes
from app.graph.workflow import init_graph, close_graph

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

    # Routes (registered before static mount so API paths take priority)
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth_routes.router, prefix=settings.API_PREFIX, tags=["Auth"])
    app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["Chat"])

    # Serve chat UI — mount last so API routes win on any overlap
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app

app = create_application()