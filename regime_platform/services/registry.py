"""
SessionRegistry — global store of active StreamSessions.

One registry per process. Accessed by all WebSocket and SSE handlers.
Thread-safe via asyncio.Lock.
"""

from __future__ import annotations
import asyncio
from typing import Optional
from .stream_session import StreamSession


class SessionRegistry:
    def __init__(self):
        self._sessions: dict[str, StreamSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        symbol: str,
        window: int = 500,
        kelly_fraction: float = 0.5,
        target_vol: float = 0.15,
        base_risk_limit: float = 0.02,
        refit_every: int = 50,
    ) -> StreamSession:
        async with self._lock:
            if symbol not in self._sessions:
                self._sessions[symbol] = StreamSession(
                    symbol=symbol,
                    window=window,
                    kelly_fraction=kelly_fraction,
                    target_vol=target_vol,
                    base_risk_limit=base_risk_limit,
                    refit_every=refit_every,
                )
            return self._sessions[symbol]

    async def get(self, symbol: str) -> Optional[StreamSession]:
        async with self._lock:
            return self._sessions.get(symbol)

    async def remove(self, symbol: str) -> None:
        async with self._lock:
            self._sessions.pop(symbol, None)

    async def list_symbols(self) -> list[str]:
        async with self._lock:
            return list(self._sessions.keys())

    async def stats(self) -> dict:
        async with self._lock:
            return {
                sym: {
                    "ready": s.is_ready,
                    "n_bars": len(s.price_buffer),
                    "refit_count": s._refit_count,
                    "tick_count": s._tick_count,
                    "last_regime": s._last_regime or "pending",
                }
                for sym, s in self._sessions.items()
            }


# Singleton — imported everywhere
registry = SessionRegistry()
