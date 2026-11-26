import sys
import logging
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger(__name__)

# libpq-style query params that hosted Postgres providers (Render, Neon,
# Supabase) put in their connection strings but asyncpg rejects as unknown
# connect() kwargs. asyncpg spells TLS `ssl`; it has no channel_binding.
_LIBPQ_ONLY_PARAMS = {"channel_binding"}

# --- DATABASE CONNECTION SANITIZER ---
def get_sanitized_url(url: str) -> str:
    """
    Forces the correct AsyncPG driver and 127.0.0.1 IP.
    This bypasses DNS resolution issues on Windows, and translates the
    libpq query params hosted-Postgres connection strings carry.
    """
    # 1. Ensure Async Driver
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if "postgresql://" in url and "postgresql+asyncpg://" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://")

    # 2. Force IP Address (Fix for [Errno 11003])
    if "@localhost" in url:
        url = url.replace("@localhost", "@127.0.0.1")

    # 3. libpq query params -> asyncpg (only touches asyncpg URLs that have a query)
    if url.startswith("postgresql+asyncpg://") and "?" in url:
        parts = urlsplit(url)
        params = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if key in _LIBPQ_ONLY_PARAMS:
                continue
            params.append(("ssl" if key == "sslmode" else key, value))
        url = urlunsplit(parts._replace(query=urlencode(params)))

    return url

# Apply the fix
FINAL_DB_URL = get_sanitized_url(settings.DATABASE_URL)

logger.debug("Connecting to: %s", FINAL_DB_URL.split("@")[-1])

# Create Engine
try:
    engine = create_async_engine(
        FINAL_DB_URL,
        echo=(settings.LOG_LEVEL == "DEBUG"),
        future=True,
        # Hosted free-tier Postgres (and Render's own network) drops idle
        # connections; without a liveness check the pool hands the next request
        # a dead one and it 500s. Costs one cheap round trip per checkout.
        pool_pre_ping=True,
    )
except Exception as e:
    logger.critical("Database engine creation failed: %s", e)
    sys.exit(1)

# Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()