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
        from ..auth.clerk_auth import CLERK_AUTH_ENABLED, verify_clerk_token

        if CLERK_AUTH_ENABLED:
            try:
                clerk_user = verify_clerk_token(token)
                # Convert ClerkTokenData → TokenData for compatibility
                return TokenData(
                    sub=clerk_user.sub,
                    role=clerk_user.role if clerk_user.role else "authenticated",
                )
            except HTTPException:
                # Clerk verification failed — try old JWT next
                pass
    except ImportError:
        pass

    # Fall back to local JWT verification
    return decode_token(token)
