import sys
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger(__name__)

# --- DATABASE CONNECTION SANITIZER ---
def get_sanitized_url(url: str) -> str:
    """
    Forces the correct AsyncPG driver and 127.0.0.1 IP.
    This bypasses DNS resolution issues on Windows.
    """
    # 1. Ensure Async Driver
    if "postgresql://" in url and "postgresql+asyncpg://" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    
    # 2. Force IP Address (Fix for [Errno 11003])
    if "@localhost" in url:
        url = url.replace("@localhost", "@127.0.0.1")
        
    return url

# Apply the fix
FINAL_DB_URL = get_sanitized_url(settings.DATABASE_URL)

logger.debug("Connecting to: %s", FINAL_DB_URL.split("@")[-1])

# Create Engine
try:
    engine = create_async_engine(
        FINAL_DB_URL,
        echo=(settings.LOG_LEVEL == "DEBUG"),
        future=True
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