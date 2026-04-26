"""
Redis Streams Persistence
═════════════════════════

Provides optional Redis-backed persistence for StreamSession price buffers.
When enabled, price history survives process restarts and can be shared
across multiple uvicorn workers.

Architecture
────────────
  • Each symbol gets a Redis Stream key: "regime:prices:{symbol}"
  • On session creation, buffer is hydrated from Redis if data exists
  • On each tick, price is appended to the Redis Stream (XADD)
  • Stream length is capped at `window` with MAXLEN (approx)
  • On session removal, the Redis key is optionally preserved

Usage
─────
  # In environment: set REDIS_URL=redis://localhost:6379
  # If REDIS_URL is not set, the layer is a no-op (in-memory only)

  from regime_platform.services.redis_persistence import RedisPersistence

  persistence = RedisPersistence()          # auto-detects REDIS_URL
  await persistence.restore(session)        # hydrate buffer from Redis
  await persistence.append_price(sym, p)   # called by StreamSession.tick()
  await persistence.delete(symbol)         # on session removal

Error policy
────────────
  All Redis operations are wrapped in try/except. A Redis failure NEVER
  raises in the tick path — it logs a warning and the system continues
  with in-memory state. Redis is persistence, not availability.
"""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .stream_session import StreamSession

log = logging.getLogger("regime.redis")

REDIS_URL = os.getenv("REDIS_URL", "")
STREAM_PREFIX = "regime:prices:"
META_PREFIX   = "regime:meta:"


class RedisPersistence:
    """
    Optional Redis Streams persistence layer.

    If REDIS_URL is not configured (or redis package unavailable),
    all methods become no-ops and `enabled` is False.
    """

    def __init__(self, redis_url: str = REDIS_URL, maxlen: int = 5000):
        self.maxlen    = maxlen
        self._redis    = None
        self._enabled  = False

        if not redis_url:
            log.info("REDIS_URL not set — price buffer persistence disabled (in-memory only)")
            return

        try:
            import redis.asyncio as aioredis
            self._client_cls = aioredis.from_url
            self._url        = redis_url
            self._enabled    = True
            log.info(f"Redis persistence configured: {redis_url}")
        except ImportError:
            log.warning("redis package not installed — persistence disabled. pip install redis")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _get_client(self):
        """Lazy connection — created on first use."""
        if self._redis is None and self._enabled:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    # ── Public API ─────────────────────────────────────────────────────────────

    async def restore(self, session: "StreamSession") -> int:
        """
        Hydrate the session's price buffer from Redis on startup.

        Returns the number of prices restored. Zero if Redis is disabled
        or no data exists for this symbol.
        """
        if not self._enabled:
            return 0
        try:
            client = await self._get_client()
            key    = f"{STREAM_PREFIX}{session.symbol}"

            # Read up to `window` most recent entries
            entries = await client.xrevrange(key, count=session.window)
            if not entries:
                return 0

            # Entries are newest-first; reverse to chronological order
            prices = [float(entry[1]["price"]) for entry in reversed(entries)]

            # Seed the session buffer directly (bypass HMM warm-up check)
            from collections import deque
            session._prices = deque(prices, maxlen=session.window)
            session._peak_price = max(prices)

            log.info(f"Restored {len(prices)} prices for {session.symbol} from Redis")
            return len(prices)

        except Exception as exc:
            log.warning(f"Redis restore failed for {session.symbol}: {exc}")
            return 0

    async def append_price(self, symbol: str, price: float, window: int = 500) -> None:
        """
        Append a price to the symbol's Redis Stream.
        MAXLEN is approximate (~10% over) for performance.
        Called on every tick — must be fast and failure-tolerant.
        """
        if not self._enabled:
            return
        try:
            client = await self._get_client()
            key    = f"{STREAM_PREFIX}{symbol}"
            await client.xadd(
                key,
                {"price": str(price)},
                maxlen=self.maxlen,
                approximate=True,
            )
        except Exception as exc:
            log.debug(f"Redis append failed for {symbol}: {exc}")

    async def save_meta(self, symbol: str, meta: dict) -> None:
        """Store session metadata (kelly_fraction, target_vol, etc.) in Redis."""
        if not self._enabled:
            return
        try:
            client = await self._get_client()
            key    = f"{META_PREFIX}{symbol}"
            await client.hset(key, mapping={k: str(v) for k, v in meta.items()})
        except Exception as exc:
            log.debug(f"Redis meta save failed for {symbol}: {exc}")

    async def load_meta(self, symbol: str) -> dict:
        """Load session metadata from Redis. Returns empty dict if not found."""
        if not self._enabled:
            return {}
        try:
            client = await self._get_client()
            key    = f"{META_PREFIX}{symbol}"
            data   = await client.hgetall(key)
            return {
                "window":          int(data.get("window", 500)),
                "kelly_fraction":  float(data.get("kelly_fraction", 0.5)),
                "target_vol":      float(data.get("target_vol", 0.15)),
                "base_risk_limit": float(data.get("base_risk_limit", 0.02)),
                "refit_every":     int(data.get("refit_every", 50)),
            }
        except Exception as exc:
            log.debug(f"Redis meta load failed for {symbol}: {exc}")
            return {}

    async def delete(self, symbol: str, preserve_history: bool = True) -> None:
        """
        Remove session metadata from Redis.
        If preserve_history=True (default), the price stream is kept so
        a re-created session can hydrate from it. Set False to fully clean up.
        """
        if not self._enabled:
            return
        try:
            client = await self._get_client()
            await client.delete(f"{META_PREFIX}{symbol}")
            if not preserve_history:
                await client.delete(f"{STREAM_PREFIX}{symbol}")
        except Exception as exc:
            log.debug(f"Redis delete failed for {symbol}: {exc}")

    async def list_persisted_symbols(self) -> list[str]:
        """Return all symbols that have persisted price streams in Redis."""
        if not self._enabled:
            return []
        try:
            client  = await self._get_client()
            pattern = f"{STREAM_PREFIX}*"
            keys    = await client.keys(pattern)
            return [k.replace(STREAM_PREFIX, "") for k in keys]
        except Exception as exc:
            log.debug(f"Redis list failed: {exc}")
            return []

    async def close(self) -> None:
        """Close the Redis connection gracefully."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass


# Singleton — shared across all sessions
persistence = RedisPersistence()
