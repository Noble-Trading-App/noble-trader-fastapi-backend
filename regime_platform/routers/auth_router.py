"""
Authentication endpoints.

  POST /auth/token   — issue a JWT (username/password or API key exchange)
  GET  /auth/me      — return current user info from token
  POST /auth/refresh — refresh a non-expired token

In production, the /auth/token endpoint should validate against a real
user store. The implementation here uses a configurable in-memory user
map (AUTH_USERS env var) for demonstration purposes.

AUTH_USERS format (env var):
  "username1:password1:role1,username2:password2:role2"
  e.g. "admin:secret123:admin,trader1:pass456:trader,readonly:pass:viewer"

If AUTH_USERS is not set, the token endpoint is disabled and returns 503.
"""

from __future__ import annotations
import os
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from ..auth.jwt_auth import (
    create_access_token, get_current_user, make_login_response,
    TokenData, AUTH_ENABLED, JWT_EXPIRE_MINS,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── In-memory user store (configurable via env) ─────────────────────────────

def _load_users() -> dict[str, dict]:
    raw = os.getenv("AUTH_USERS", "")
    users = {}
    for entry in raw.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3:
            username, password, role = parts
            users[username] = {"password": password, "role": role}
    return users

_USERS = _load_users()


# ── Response models ──────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str
    expires_in:   int
    sub:          str
    role:         str


class UserInfo(BaseModel):
    sub:          str
    role:         str
    auth_enabled: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue a JWT access token",
)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Standard OAuth2 password flow.

    Supply `username` and `password` as form fields.
    Returns a Bearer JWT valid for `JWT_EXPIRE_MINS` minutes.

    Configure users via the `AUTH_USERS` environment variable:
    ```
    AUTH_USERS="admin:mysecret:admin,trader:pass123:trader"
    ```

    Roles: `admin` (full access) · `trader` (read + write) · `viewer` (read-only)
    """
    if not AUTH_ENABLED:
        # Dev mode — issue a token without credential check
        return TokenResponse(**make_login_response(sub=form.username, role="admin"))

    if not _USERS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AUTH_USERS environment variable not configured on server.",
        )

    user = _USERS.get(form.username)
    if not user or user["password"] != form.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(**make_login_response(sub=form.username, role=user["role"]))


@router.get(
    "/me",
    response_model=UserInfo,
    summary="Return current user info decoded from token",
)
async def me(user: TokenData = Depends(get_current_user)):
    """
    Returns the identity and role extracted from the Bearer JWT.
    Useful for verifying a token is valid and checking its role.
    """
    return UserInfo(sub=user.sub, role=user.role, auth_enabled=AUTH_ENABLED)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh a (non-expired) token",
)
async def refresh(user: TokenData = Depends(get_current_user)):
    """
    Issues a fresh token with a new expiry, keeping the same subject and role.
    Only works if the current token is still valid (not yet expired).
    """
    return TokenResponse(**make_login_response(sub=user.sub, role=user.role))
