from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import settings

# Shared limiter — imported by server.py (to register exception handler)
# and by route modules (to apply @limiter.limit decorators).
# Default storage is memory:// (no Redis needed in dev).
# Set REDIS_URL=redis://127.0.0.1:6379/0 in .env to enable shared Redis storage.
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.REDIS_URL)
