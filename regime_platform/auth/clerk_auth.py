"""
Clerk JWT Authentication
═══════════════════════

Provides Clerk-based JWT authentication for FastAPI.
Uses Clerk's publishable and secret keys from environment variables.

Configuration (environment variables)
──────────────────────────────────────
  CLERK_PUBLISHABLE_KEY  — Clerk publishable key (for client-side)
  CLERK_SECRET_KEY      — Clerk secret key (for server-side verification)
  CLERK_JWT_KEY         — Optional: specific JWT key ID to use

Clerk Token Verification
─────────────────────────
Clerk issues JWT tokens that can be verified using the CLERK_SECRET_KEY.
Tokens contain standard claims (sub, exp, iat) and custom Clerk claims.

Usage
─────
  # Verify a Clerk JWT token
  from regime_platform.auth.clerk_auth import verify_clerk_token, get_clerk_user

  token = "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..."
  user_data = verify_clerk_token(token)

  # Or use as FastAPI dependency
  from regime_platform.auth.clerk_auth import get_current_clerk_user

  @router.get("/protected")
  async def protected_route(user = Depends(get_current_clerk_user)):
      return {"user_id": user.sub, "role": user.role}
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning(
        "PyJWT not installed. Clerk auth will not work. Install with: pip install PyJWT[crypto]"
    )

log = logging.getLogger("regime.auth.clerk")

# ── Configuration ─────────────────────────────────────────────────────────────

CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")
CLERK_JWT_KEY = os.getenv("CLERK_JWT_KEY", None)  # Optional: specific key ID

# Clerk uses RS256 algorithm for signing JWTs
CLERK_JWT_ALGORITHM = "RS256"

# Enable/disable Clerk authentication
CLERK_AUTH_ENABLED = bool(CLERK_SECRET_KEY)

if not CLERK_AUTH_ENABLED:
    log.warning(
        "CLERK_SECRET_KEY is not set. Clerk authentication will be disabled. "
        "Set CLERK_SECRET_KEY in .env.local to enable Clerk auth."
    )


# ── Token Data Model ────────────────────────────────────────────────────────


class ClerkTokenData:
    """
    Represents a verified Clerk user from a JWT token.

    Attributes:
        sub: User ID (Clerk user ID)
        role: User role (extracted from token or default)
        email: User email address
        first_name: User's first name
        last_name: User's last name
        username: User's username
        exp: Token expiration timestamp
        iat: Token issued at timestamp
        claims: Full token payload
    """

    def __init__(
        self,
        sub: str,
        role: str = "viewer",
        email: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        username: Optional[str] = None,
        exp: Optional[datetime] = None,
        iat: Optional[datetime] = None,
        claims: Optional[dict] = None,
    ):
        self.sub = sub
        self.role = role
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.username = username
        self.exp = exp
        self.iat = iat
        self.claims = claims or {}

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == "admin"

    @property
    def can_write(self) -> bool:
        """Check if user has write access (admin or trader)."""
        return self.role in ("admin", "trader")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON responses."""
        return {
            "sub": self.sub,
            "role": self.role,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "username": self.username,
        }


# ── Token Verification ──────────────────────────────────────────────────────


def _get_clerk_jwks() -> dict:
    """
    Fetch Clerk's JWKS (JSON Web Key Set) from their public endpoint.

    Note: For production, consider caching this response.
    """
    import httpx

    # Extract domain from publishable key or use default
    # Clerk publishable key format: pk_test_... or pk_live_...
    # We can extract the environment, but for JWKS we need the domain

    # For now, use Clerk's default JWKS endpoint
    # In production, you might want to use your specific Clerk instance URL
    jwks_url = "https://jwks.clerk.accounts.dev/.well-known/jwks.json"

    try:
        response = httpx.get(jwks_url, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log.error(f"Failed to fetch Clerk JWKS: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch Clerk JWKS",
        )


def _find_matching_key(jwks: dict, kid: Optional[str] = None) -> Optional[dict]:
    """
    Find a matching key in JWKS by key ID (kid).
    If no kid specified, use the first RS256 key.
    """
    keys = jwks.get("keys", [])

    if kid:
        for key in keys:
            if key.get("kid") == kid:
                return key
        return None

    # Return first RS256 key if no specific kid
    for key in keys:
        if key.get("kty") == "RSA" and key.get("use") == "sig":
            return key

    return None


def _verify_token_with_jwks(token: str, jwks: dict, kid: Optional[str] = None) -> dict:
    """
    Verify a JWT token using JWKS keys.
    """
    import jwt
    from jwt import PyJWK, PyJWKClient

    jwk = _find_matching_key(jwks, kid)
    if not jwk:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"No matching key found in JWKS for kid: {kid}",
        )

    # Create PyJWK from the key
    public_key = PyJWK.from_dict(jwk)

    try:
        payload = jwt.decode(
            token,
            public_key.key,
            algorithms=[CLERK_JWT_ALGORITHM],
            options={"verify_exp": True, "verify_iat": True},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_clerk_token(token: str) -> ClerkTokenData:
    """
    Verify a Clerk-issued JWT token.

    This function:
    1. Decodes the JWT header to get the key ID (kid)
    2. Fetches Clerk's JWKS
    3. Finds the matching key
    4. Verifies the token signature and expiration
    5. Extracts user information

    Parameters
    ----------
    token : str
        The JWT token string

    Returns
    -------
    ClerkTokenData
        Verified user data from the token

    Raises
    ------
    HTTPException
        If token is invalid, expired, or verification fails
    """
    if not CLERK_AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clerk authentication is not enabled",
        )

    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT is not installed",
        )

    try:
        import jwt

        # Decode header to get kid without verification
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        # Fetch JWKS
        jwks = _get_clerk_jwks()

        # Verify token
        payload = _verify_token_with_jwks(token, jwks, kid)

        # Extract user information
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim",
            )

        # Extract Clerk-specific claims
        # Clerk tokens can have various claims depending on token type
        # Common claims: sid (session ID), azp (authorized party), etc.

        # Try to extract role from token metadata or use default
        # Clerk doesn't natively have roles in tokens, but you can add
        # custom claims when creating tokens via Clerk API
        role = payload.get("role", "viewer")

        # Extract user metadata if available in token
        email = payload.get("email") or payload.get("email_addresses", [{}])[0].get(
            "email_address"
        )
        first_name = payload.get("first_name") or payload.get("firstName")
        last_name = payload.get("last_name") or payload.get("lastName")
        username = payload.get("username")

        # Convert timestamps
        exp = payload.get("exp")
        iat = payload.get("iat")

        if exp:
            exp = datetime.fromtimestamp(exp, tz=timezone.utc)
        if iat:
            iat = datetime.fromtimestamp(iat, tz=timezone.utc)

        return ClerkTokenData(
            sub=str(sub),
            role=str(role),
            email=str(email) if email else None,
            first_name=str(first_name) if first_name else None,
            last_name=str(last_name) if last_name else None,
            username=str(username) if username else None,
            exp=exp,
            iat=iat,
            claims=payload,
        )

    except jwt.DecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies ────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


async def get_current_clerk_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> ClerkTokenData:
    """
    FastAPI dependency for Clerk JWT authentication.

    Extracts and verifies a Clerk JWT from the Authorization header.

    Usage:
        @router.get("/protected")
        async def protected_route(user = Depends(get_current_clerk_user)):
            return {"user_id": user.sub, "email": user.email}

    Returns
    -------
    ClerkTokenData
        Verified user data

    Raises
    ------
    HTTPException
        If authentication fails
    """
    if not CLERK_AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clerk authentication is not enabled",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization scheme must be Bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    return verify_clerk_token(token)


async def get_optional_clerk_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[ClerkTokenData]:
    """
    FastAPI dependency for optional Clerk JWT authentication.

    Returns None if no token is provided, otherwise verifies and returns user.

    Usage:
        @router.get("/public-or-private")
        async def route(user = Depends(get_optional_clerk_user)):
            if user:
                return {"message": f"Hello, {user.email}"}
            return {"message": "Hello, guest"}
    """
    if not CLERK_AUTH_ENABLED:
        return None

    if credentials is None:
        return None

    if credentials.scheme.lower() != "bearer":
        return None

    try:
        token = credentials.credentials
        return verify_clerk_token(token)
    except HTTPException:
        return None


async def require_clerk_admin(
    user: ClerkTokenData = Depends(get_current_clerk_user),
) -> ClerkTokenData:
    """
    Dependency that requires admin role from Clerk-authenticated user.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user


async def require_clerk_write(
    user: ClerkTokenData = Depends(get_current_clerk_user),
) -> ClerkTokenData:
    """
    Dependency that requires write access (admin or trader role) from Clerk-authenticated user.
    """
    if not user.can_write:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write access required (admin or trader role)",
        )
    return user


# ── Token Creation (Server-side) ─────────────────────────────────────────────


def create_clerk_compatible_token(
    sub: str,
    role: str = "viewer",
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    username: Optional[str] = None,
    expires_in_minutes: int = 60,
) -> str:
    """
    Create a JWT token compatible with Clerk verification.

    Note: This uses the CLERK_SECRET_KEY for signing, which allows
    the token to be verified by Clerk's systems. However, typically
    you would use Clerk's API to create tokens, not create them locally.

    This function is provided for cases where you need to issue
    server-side tokens that can be verified by the same Clerk setup.

    Parameters
    ----------
    sub : str
        User ID (subject)
    role : str
        User role (admin, trader, viewer)
    email : str, optional
        User email address
    first_name : str, optional
        User's first name
    last_name : str, optional
        User's last name
    username : str, optional
        User's username
    expires_in_minutes : int
        Token expiration in minutes

    Returns
    -------
    str
        Signed JWT token string
    """
    if not CLERK_AUTH_ENABLED:
        raise RuntimeError("Clerk authentication is not enabled")

    if not CLERK_SECRET_KEY:
        raise RuntimeError("CLERK_SECRET_KEY is not configured")

    if not JWT_AVAILABLE:
        raise RuntimeError("PyJWT is not installed")

    from datetime import datetime, timedelta, timezone

    import jwt

    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=expires_in_minutes)

    payload = {
        "sub": sub,
        "role": role,
        "exp": int(exp.timestamp()),
        "iat": int(now.timestamp()),
    }

    if email:
        payload["email"] = email
    if first_name:
        payload["first_name"] = first_name
    if last_name:
        payload["last_name"] = last_name
    if username:
        payload["username"] = username

    # Sign with Clerk secret key
    token = jwt.encode(
        payload,
        CLERK_SECRET_KEY,
        algorithm=CLERK_JWT_ALGORITHM,
    )

    return token


# ── Webhook Verification ────────────────────────────────────────────────────


async def verify_clerk_webhook(
    request: Request,
    secret: Optional[str] = None,
) -> dict:
    """
    Verify a Clerk webhook request.

    Clerk sends webhooks with a signature in the Svix-Signature header.
    This function verifies that the webhook came from Clerk.

    Parameters
    ----------
    request : Request
        FastAPI Request object
    secret : str, optional
        Webhook secret. If not provided, uses CLERK_SECRET_KEY

    Returns
    -------
    dict
        Parsed webhook payload

    Raises
    ------
    HTTPException
        If webhook verification fails
    """
    import hashlib
    import hmac

    webhook_secret = secret or CLERK_SECRET_KEY

    if not webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret not configured",
        )

    svix_signature = request.headers.get("Svix-Signature")
    if not svix_signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Svix-Signature header",
        )

    svix_id = request.headers.get("Svix-Id")
    svix_timestamp = request.headers.get("Svix-Timestamp")

    if not svix_id or not svix_timestamp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Svix-Id or Svix-Timestamp header",
        )

    # Read request body
    body = await request.body()

    # Construct the signed payload
    signed_payload = f"{svix_id}.{svix_timestamp}.{body.decode('utf-8')}"

    # Calculate expected signature
    expected_signature = hmac.new(
        webhook_secret.encode("utf-8"),
        signed_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Verify signature (Svix-Signature format: "v1,{signature}")
    provided_signature = (
        svix_signature.split(",v1=")[1] if ",v1=" in svix_signature else svix_signature
    )

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # Parse and return payload
    import json
    return json.loads(body.decode("utf-8"))

