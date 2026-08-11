import logging

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.database import get_db
from app.infrastructure.models import User
from app.core.security import hash_password, verify_password, create_access_token
from app.core.config import settings
from app.api.limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    credential: str  # the ID token from Google Identity Services' callback


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


class AuthConfigResponse(BaseModel):
    google_client_id: str  # "" means Google Sign-In isn't configured


@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalars().first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(sub=user.id),
        user_id=user.id,
    )


@router.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalars().first()
    if not user or not user.hashed_password or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=create_access_token(sub=user.id),
        user_id=user.id,
    )


@router.get("/auth/config", response_model=AuthConfigResponse)
async def auth_config():
    """Public — lets the frontend know whether to render the Google button
    at all, without hardcoding the client ID into a static JS file that
    would then differ between local dev and prod. The client ID itself
    isn't secret (Google embeds it in every browser-issued ID token
    regardless); this just keeps ONE source of truth for it, in .env."""
    return AuthConfigResponse(google_client_id=settings.GOOGLE_OAUTH_CLIENT_ID)


async def _verify_google_credential(credential: str) -> dict:
    """Runs in a thread: verify_oauth2_token makes a blocking HTTP call to
    fetch Google's public signing certs (cached by the underlying session
    after the first call, but still a real network round-trip that must not
    block the event loop — same reasoning as RAGService's asyncio.to_thread
    use for the cross-encoder)."""
    import asyncio
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    return await asyncio.to_thread(
        google_id_token.verify_oauth2_token,
        credential,
        google_requests.Request(),
        settings.GOOGLE_OAUTH_CLIENT_ID,
    )


@router.post("/auth/google", response_model=TokenResponse)
@limiter.limit("10/minute")
async def google_auth(request: Request, payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    if not settings.GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")

    try:
        idinfo = await _verify_google_credential(payload.credential)
    except Exception as e:
        # Wrong audience, expired token, bad signature, or a network failure
        # fetching Google's certs all land here — none of these should leak
        # detail to the client, but they're worth knowing about server-side.
        logger.warning("Google credential verification failed: %s", e)
        raise HTTPException(status_code=401, detail="Invalid Google credential")

    # Google verifies the address itself before setting this — required
    # before trusting the claim for account lookup/creation, since an
    # unverified email on the token could belong to someone else entirely.
    if not idinfo.get("email_verified"):
        raise HTTPException(status_code=401, detail="Google account email is not verified")

    google_sub = idinfo["sub"]
    email = idinfo["email"]

    result = await db.execute(select(User).where(User.google_sub == google_sub))
    user = result.scalars().first()

    if not user:
        # Not linked yet. If a password account already exists with this
        # (Google-verified) email, link Google onto it rather than creating
        # a second, disconnected account for the same person.
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        if user:
            user.google_sub = google_sub
            db.add(user)
            await db.commit()
        else:
            user = User(email=email, google_sub=google_sub, hashed_password=None)
            db.add(user)
            await db.commit()
            await db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(sub=user.id),
        user_id=user.id,
    )
