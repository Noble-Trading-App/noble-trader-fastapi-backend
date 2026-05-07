"""
JWT Authentication
══════════════════

Provides JWT-based authentication for:
  • HTTP endpoints  — FastAPI Depends(get_current_user)
  • WebSocket       — Token passed as ?token= query param
  • API key fallback — X-API-Key header for service-to-service calls

Configuration (environment variables)
──────────────────────────────────────
  JWT_SECRET_KEY   — signing secret (required; no default for security)
  JWT_ALGORITHM    — default "HS256"
  JWT_EXPIRE_MINS  — token lifetime in minutes (default 60)
  API_KEYS         — comma-separated valid API keys (optional)
  AUTH_ENABLED     — set "false" to disable auth entirely (dev mode)

Usage
─────
  # Protect an HTTP endpoint
  @router.post("/endpoint")
  async def handler(user=Depends(get_current_user)):
      ...

  # Protect a WebSocket endpoint
  @router.websocket("/ws/{symbol}")
  async def ws(websocket, symbol, token=Query(None)):
      user = await ws_auth(websocket, token)
      ...

  # Issue a token (login endpoint)
  token = create_access_token({"sub": "trader1", "role": "viewer"})
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

log = logging.getLogger("regime.auth")

# ── Configuration ─────────────────────────────────────────────────────────────

JWT_SECRET_KEY  = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM   = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINS = int(os.getenv("JWT_EXPIRE_MINS", "60"))
API_KEYS        = set(filter(None, os.getenv("API_KEYS", "").split(",")))
AUTH_ENABLED    = os.getenv("AUTH_ENABLED", "true").lower() != "false"

if AUTH_ENABLED and not JWT_SECRET_KEY:
    log.warning(
        "JWT_SECRET_KEY is not set. Authentication will reject all JWT tokens. "
        "Set JWT_SECRET_KEY env var or set AUTH_ENABLED=false for development."
    )


# ── Token data model ──────────────────────────────────────────────────────────

class TokenData:
    def __init__(self, sub: str, role: str = "viewer", exp: Optional[datetime] = None):
        self.sub  = sub    # subject / user ID
        self.role = role   # "admin" | "trader" | "viewer"
        self.exp  = exp

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_write(self) -> bool:
        return self.role in ("admin", "trader")


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Parameters
    ──────────
    data           Payload dict. Should include "sub" (subject/user).
    expires_delta  Token lifetime. Defaults to JWT_EXPIRE_MINS.

    Returns
    ───────
    Signed JWT string.
    """
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed: pip install python-jose[cryptography]")

    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured")

    payload = data.copy()
    expire  = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_EXPIRE_MINS)
    )
    payload.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Raises
    ──────
    HTTPException 401 if token is invalid, expired, or secret is missing.
    """
    try:
        from jose import jwt, JWTError
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="python-jose not installed on server",
        )

    if not JWT_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT authentication not configured on server",
        )

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        sub  = payload.get("sub")
        role = payload.get("role", "viewer")
        exp  = payload.get("exp")
        if sub is None:
            raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
        return TokenData(sub=str(sub), role=str(role))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI security schemes ──────────────────────────────────────────────────

_bearer     = HTTPBearer(auto_error=False)
_api_key_hdr = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    api_key: Optional[str] = Depends(_api_key_hdr),
) -> TokenData:
    """
    FastAPI dependency for HTTP endpoint authentication.

    Accepts:
      - Bearer JWT token in Authorization header
      - X-API-Key header (service-to-service)

    If AUTH_ENABLED=false, returns a synthetic admin user (dev mode).

    Usage:
        @router.post("/endpoint")
        async def handler(user: TokenData = Depends(get_current_user)):
            ...
    """
    if not AUTH_ENABLED:
        return TokenData(sub="dev", role="admin")

    # API key check first (simpler, faster)
    if api_key and api_key in API_KEYS:
        return TokenData(sub=f"apikey:{api_key[:8]}...", role="trader")

    # JWT check
    if credentials and credentials.scheme.lower() == "bearer":
        return decode_token(credentials.credentials)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide a Bearer JWT or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_write(user: TokenData = Depends(get_current_user)) -> TokenData:
    """Dependency that additionally requires trader or admin role."""
    if not user.can_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role '{user.role}' does not have write access.",
        )
    return user


async def require_admin(user: TokenData = Depends(get_current_user)) -> TokenData:
    """Dependency that requires admin role."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )
    return user


# ── Unified auth (Clerk JWT → old JWT → API key) ─────────────────────────────

async def get_authed_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    api_key: Optional[str] = Depends(_api_key_hdr),
) -> TokenData:
    """
    Unified auth dependency that tries Clerk JWT first, then falls back
    to the traditional JWT/API-key auth.

    Priority order:
      1. Bearer token → try Clerk JWKS verification
      2. Bearer token → try local JWT_SECRET_KEY verification
      3. X-API-Key header
      4. AUTH_ENABLED=false → dev mode (synthetic admin)

    This allows both Clerk-issued tokens AND locally-signed tokens to work,
    making the transition seamless.
    """
    # Dev mode bypass
    if not AUTH_ENABLED:
        return TokenData(sub="dev", role="admin")

    # API key check (fastest)
    if api_key and api_key in API_KEYS:
        return TokenData(sub=f"apikey:{api_key[:8]}...", role="trader")

    # No credentials at all
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide a Bearer JWT or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Try Clerk JWT verification first (if Clerk is enabled)
    try:
        from ..auth.clerk_auth import verify_clerk_token, CLERK_AUTH_ENABLED
        if CLERK_AUTH_ENABLED:
            try:
                clerk_user = verify_clerk_token(token)
                # Convert ClerkTokenData → TokenData for compatibility
                return TokenData(
                    sub=clerk_user.sub,
                    role=clerk_user.role if clerk_user.role else "authenticated",
                )
            except HTTPException as clerk_err:
                # Clerk is enabled but verification failed.
                # If JWT_SECRET_KEY is also configured, try old JWT as fallback.
                # Otherwise, surface the Clerk error (don't silently fall through
                # to decode_token which would give a confusing "not configured" msg).
                if JWT_SECRET_KEY:
                    log.warning(
                        "Clerk JWT verification failed (%s), falling back to local JWT",
                        clerk_err.detail,
                    )
                    return decode_token(token)
                # No local JWT configured — Clerk is the only auth method; re-raise
                raise
    except ImportError:
        log.warning("clerk_auth module not importable; skipping Clerk verification")

    # Fall back to local JWT verification (only reached if Clerk is not enabled)
    return decode_token(token)


# ── WebSocket authentication ──────────────────────────────────────────────────

async def ws_auth(
    websocket: WebSocket,
    token: Optional[str] = None,
    api_key: Optional[str] = None,
) -> TokenData:
    """
    Authenticate a WebSocket connection before accepting it.

    Checks (in order):
      1. ?token=<jwt> query parameter
      2. ?api_key=<key> query parameter
      3. X-API-Key header
      4. AUTH_ENABLED=false → dev mode, always accept

    Closes the WebSocket with 4001 (Unauthorised) if authentication fails.

    Usage:
        @router.websocket("/ws/{symbol}")
        async def ws_handler(
            websocket: WebSocket,
            symbol: str,
            token: Optional[str] = Query(None),
            api_key: Optional[str] = Query(None),
        ):
            user = await ws_auth(websocket, token, api_key)
            await websocket.accept()
            ...
    """
    if not AUTH_ENABLED:
        return TokenData(sub="dev", role="admin")

    # Query-param API key
    if api_key and api_key in API_KEYS:
        return TokenData(sub=f"apikey:{api_key[:8]}...", role="trader")

    # Query-param JWT / Clerk JWT
    if token:
        try:
            # Try Clerk verification first
            try:
                from ..auth.clerk_auth import verify_clerk_token, CLERK_AUTH_ENABLED
                if CLERK_AUTH_ENABLED:
                    try:
                        clerk_user = verify_clerk_token(token)
                        return TokenData(
                            sub=clerk_user.sub,
                            role=clerk_user.role if clerk_user.role else "authenticated",
                        )
                    except HTTPException:
                        if not JWT_SECRET_KEY:
                            raise
                        # Fall back to local JWT if secret is configured
            except ImportError:
                pass
            return decode_token(token)
        except HTTPException:
            await websocket.close(code=4001, reason="Invalid token")
            raise

    # Header API key
    hdr_key = websocket.headers.get("x-api-key")
    if hdr_key and hdr_key in API_KEYS:
        return TokenData(sub=f"apikey:{hdr_key[:8]}...", role="trader")

    await websocket.close(code=4001, reason="Unauthorised")
    raise HTTPException(status_code=401, detail="WebSocket authentication failed")


# ── Login endpoint helper ─────────────────────────────────────────────────────

def make_login_response(sub: str, role: str = "viewer") -> dict:
    """
    Utility for a /auth/token endpoint to return a standard token response.

    Usage:
        @router.post("/auth/token")
        async def login(form: OAuth2PasswordRequestForm = Depends()):
            # Validate credentials here ...
            return make_login_response(sub=form.username, role="trader")
    """
    token = create_access_token({"sub": sub, "role": role})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   JWT_EXPIRE_MINS * 60,
        "sub":          sub,
        "role":         role,
    }
