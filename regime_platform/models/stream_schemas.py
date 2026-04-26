"""Streaming-specific Pydantic schemas."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class SeedRequest(BaseModel):
    """Seed a symbol session with historical prices before streaming."""
    symbol: str
    prices: list[float] = Field(..., min_length=81)
    window: int = Field(default=500, ge=81, le=5000)
    kelly_fraction: float = Field(default=0.5, ge=0.1, le=1.0)
    target_vol: float = Field(default=0.15, gt=0, le=1.0)
    base_risk_limit: float = Field(default=0.02, gt=0, le=0.5)
    refit_every: int = Field(default=50, ge=5, le=500)


class TickIngest(BaseModel):
    """Single price tick pushed via HTTP POST (for REST-based ingestion)."""
    symbol: str
    price: float = Field(..., gt=0)
    ts: Optional[float] = None   # unix timestamp; auto-filled if omitted


class TickResponse(BaseModel):
    """Regime+sizing+risk snapshot returned for a single tick."""
    symbol: str
    ts: float
    price: float
    n_bars: int
    regime_label: str
    vol_state: str
    trend_state: str
    vol_probs: dict[str, float]
    trend_probs: dict[str, float]
    confidence: float
    risk_multiplier: float
    recommended_f: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    suggested_stop: float
    suggested_tp: float
    regime_changed: bool
    alert: Optional[str]
    refit_count: int


class SessionStatus(BaseModel):
    symbol: str
    ready: bool
    n_bars: int
    refit_count: int
    tick_count: int
    last_regime: str


class WsMessage(BaseModel):
    """Envelope for WebSocket messages from client → server."""
    type: str        # "tick" | "seed" | "subscribe" | "ping"
    symbol: str
    price: Optional[float] = None
    prices: Optional[list[float]] = None
    ts: Optional[float] = None
