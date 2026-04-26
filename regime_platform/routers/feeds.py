"""
Feed adapter management endpoints.

Manages live OHLCV feed adapters (Alpaca, Binance, IB).
Adapters stream bars directly into seeded StreamSessions.

Note: Adapter tasks run as background asyncio tasks tied to the
app lifespan. They are not persisted across restarts — reconfigure
on each startup (or use the lifespan hook in main_v4.py).
"""

from __future__ import annotations

import asyncio
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional, Literal

from ..adapters.feed_adapters import (
    FeedManager, AlpacaFeedAdapter, BinanceFeedAdapter, IBFeedAdapter
)
from ..auth.jwt_auth import require_write, TokenData

router  = APIRouter(prefix="/feeds", tags=["Feed Adapters"])
log     = logging.getLogger("regime.feeds")

# Global feed manager singleton
_manager: Optional[FeedManager] = None
_manager_task: Optional[asyncio.Task] = None


# ── Request / Response models ─────────────────────────────────────────────────

class FeedConfig(BaseModel):
    source:   Literal["alpaca", "binance", "ib"] = Field(..., description="Data source")
    symbols:  list[str]                           = Field(..., min_length=1)
    bar_size: str = Field(default="1Min",  description="Bar interval (Alpaca: '1Min', Binance: '1m', IB: '5 secs')")
    feed:     str = Field(default="sip",   description="Alpaca feed type: 'sip' or 'iex'")


class FeedStatusResponse(BaseModel):
    running:   bool
    adapters:  list[dict]
    total_bars: dict[str, int]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_adapter(cfg: FeedConfig):
    if cfg.source == "alpaca":
        return AlpacaFeedAdapter(symbols=cfg.symbols, bar_size=cfg.bar_size, feed=cfg.feed)
    elif cfg.source == "binance":
        return BinanceFeedAdapter(symbols=cfg.symbols, interval=cfg.bar_size)
    elif cfg.source == "ib":
        return IBFeedAdapter(symbols=cfg.symbols, bar_size=cfg.bar_size)
    raise ValueError(f"Unknown source: {cfg.source}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=FeedStatusResponse,
    summary="Get feed adapter status",
)
async def feed_status(user: TokenData = Depends(require_write)):
    """Returns running state and per-symbol bar counts for all active feed adapters."""
    global _manager, _manager_task
    if _manager is None:
        return FeedStatusResponse(running=False, adapters=[], total_bars={})
    done = _manager_task is None or _manager_task.done()
    stats = _manager.stats
    return FeedStatusResponse(
        running=not done,
        adapters=stats.get("adapters", []),
        total_bars=stats.get("total_bars", {}),
    )


@router.post(
    "/start",
    summary="Start live feed adapters",
)
async def start_feeds(
    feeds:      list[FeedConfig],
    background: BackgroundTasks,
    user:       TokenData = Depends(require_write),
):
    """
    Start one or more live feed adapters.

    Symbols must be seeded via `POST /stream/seed` before starting feeds —
    bars for unseeded symbols are silently dropped.

    Adapters reconnect automatically on disconnect (exponential backoff).
    Feeds run as background tasks until `POST /feeds/stop` is called.
    """
    global _manager, _manager_task

    if _manager_task and not _manager_task.done():
        raise HTTPException(status_code=409, detail="Feeds already running. Call /feeds/stop first.")

    _manager = FeedManager()
    for cfg in feeds:
        try:
            adapter = _build_adapter(cfg)
            _manager.add(adapter)
            log.info(f"Registered {cfg.source} adapter for {cfg.symbols}")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid feed config: {e}")

    # Launch as background asyncio task
    loop = asyncio.get_event_loop()
    _manager_task = loop.create_task(_manager.start())

    return {
        "status":   "started",
        "adapters": [{"source": f.source, "symbols": f.symbols} for f in feeds],
        "note":     "Bars routed to seeded sessions. Check /feeds/status for progress.",
    }


@router.post(
    "/stop",
    summary="Stop all running feed adapters",
)
async def stop_feeds(user: TokenData = Depends(require_write)):
    """Gracefully stops all feed adapter tasks."""
    global _manager, _manager_task
    if _manager is None or (_manager_task and _manager_task.done()):
        return {"status": "not_running"}
    await _manager.stop()
    _manager_task = None
    return {"status": "stopped"}
