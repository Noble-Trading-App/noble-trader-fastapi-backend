"""
Authentication module exports.

This module provides both JWT and Clerk authentication for the platform.
"""

from .clerk_auth import (
    CLERK_AUTH_ENABLED,
    CLERK_PUBLISHABLE_KEY,
    CLERK_SECRET_KEY,
    ClerkTokenData,
    create_clerk_compatible_token,
    get_current_clerk_user,
    get_optional_clerk_user,
    require_clerk_admin,
    require_clerk_write,
    verify_clerk_token,
    verify_clerk_webhook,
)
from .jwt_auth import (
    API_KEYS,
    AUTH_ENABLED,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINS,
    JWT_SECRET_KEY,
    TokenData,
    create_access_token,
    decode_token,
    get_current_user,
    make_login_response,
    require_admin,
    require_write,
    ws_auth,
)

__all__ = [
    # JWT Auth
    "TokenData",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "require_write",
    "require_admin",
    "ws_auth",
    "make_login_response",
    "JWT_SECRET_KEY",
    "JWT_ALGORITHM",
    "JWT_EXPIRE_MINS",
    "API_KEYS",
    "AUTH_ENABLED",
    # Clerk Auth
    "ClerkTokenData",
    "verify_clerk_token",
    "get_current_clerk_user",
    "get_optional_clerk_user",
    "require_clerk_admin",
    "require_clerk_write",
    "create_clerk_compatible_token",
    "verify_clerk_webhook",
    "CLERK_PUBLISHABLE_KEY",
    "CLERK_SECRET_KEY",
    "CLERK_AUTH_ENABLED",
]
