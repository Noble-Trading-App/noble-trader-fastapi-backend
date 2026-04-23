"""
REST streaming endpoints.

  POST /stream/seed/{symbol}     — seed historical prices
  POST /stream/tick              — ingest a single tick (returns tick JSON)
  POST /stream/ticks             — batch ingest multiple ticks
  GET  /stream/sessions          — list active sessions + stats
  GET  /stream/session/{symbol}  — single session status
  DELETE /stream/session/{symbol}— remove session
"""

from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException

from ..models.stream_schemas import SeedRequest, TickIngest, TickResponse, SessionStatus
from ..services.registry import registry

router = APIRouter(prefix="/stream", tags=["Live Stream"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tick_to_response(tick) -> TickResponse:
    return TickResponse(
        symbol=tick.symbol,
        ts=tick.ts,
        price=tick.price,
        n_bars=tick.n_bars,
        regime_label=tick.regime_label,
        vol_state=tick.vol_state,
        trend_state=tick.trend_state,
        vol_probs=dict(zip(["low", "medium", "high"], tick.vol_probs)),
        trend_probs=dict(zip(["bear", "neutral", "bull"], tick.trend_probs)),
        confidence=tick.confidence,
        risk_multiplier=tick.risk_multiplier,
        recommended_f=tick.recommended_f,
        sharpe_ratio=tick.sharpe_ratio,
        var_95=tick.var_95,
        cvar_95=tick.cvar_95,
        suggested_stop=tick.suggested_stop,
        suggested_tp=tick.suggested_tp,
        regime_changed=tick.regime_changed,
        alert=tick.alert,
        refit_count=tick.refit_count,
    )


# ── Seed ──────────────────────────────────────────────────────────────────────

@router.post(
    "/seed",
    summary="Seed a symbol session with historical prices",
    response_model=SessionStatus,
)
async def seed_session(req: SeedRequest):
    """
    Creates (or resets) a streaming session for `symbol` and pre-loads
    historical prices so the HMM can be fitted immediately. Must be called
    before streaming ticks for a new symbol.
    """
    session = await registry.get_or_create(
        symbol=req.symbol,
        window=req.window,
        kelly_fraction=req.kelly_fraction,
        target_vol=req.target_vol,
        base_risk_limit=req.base_risk_limit,
        refit_every=req.refit_every,
    )
    await session.seed(req.prices)
    return SessionStatus(
        symbol=req.symbol,
        ready=session.is_ready,
        n_bars=len(session.price_buffer),
        refit_count=session._refit_count,
        tick_count=session._tick_count,
        last_regime=session._last_regime or "pending",
    )


# ── Single tick ───────────────────────────────────────────────────────────────

@router.post(
    "/tick",
    response_model=TickResponse | dict,
    summary="Ingest one price tick and receive regime snapshot",
)
async def ingest_tick(req: TickIngest):
    """
    Push a single price tick. Returns a full `TickResponse` once the session
    has enough bars; returns `{"status": "buffering", "n_bars": N}` while warming up.

    The session must have been seeded first via `POST /stream/seed`.
    """
    session = await registry.get(req.symbol)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"No session for '{req.symbol}'. Seed it first via POST /stream/seed.",
        )
    tick = await session.tick(req.price, req.ts or time.time())
    if tick is None:
        return {"status": "buffering", "n_bars": len(session.price_buffer)}
    return _tick_to_response(tick)


# ── Batch ticks ───────────────────────────────────────────────────────────────

@router.post(
    "/ticks",
    response_model=list[TickResponse | dict],
    summary="Ingest a batch of price ticks",
)
async def ingest_ticks(symbol: str, prices: list[float], timestamps: list[float] | None = None):
    """
    Replay or batch-ingest multiple ticks. Returns a snapshot for each tick
    (or a buffering stub when warming up). Useful for backtesting / replay.
    """
    session = await registry.get(symbol)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session for '{symbol}'.")

    results = []
    for i, price in enumerate(prices):
        ts   = timestamps[i] if timestamps and i < len(timestamps) else time.time()
        tick = await session.tick(price, ts)
        results.append(_tick_to_response(tick) if tick else {"status": "buffering", "n_bars": len(session.price_buffer)})
    return results


# ── Session management ────────────────────────────────────────────────────────

@router.get("/sessions", summary="List all active streaming sessions")
async def list_sessions():
    return await registry.stats()


@router.get("/session/{symbol}", response_model=SessionStatus, summary="Get session status")
async def get_session(symbol: str):
    session = await registry.get(symbol)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session for '{symbol}'.")
    return SessionStatus(
        symbol=symbol,
        ready=session.is_ready,
        n_bars=len(session.price_buffer),
        refit_count=session._refit_count,
        tick_count=session._tick_count,
        last_regime=session._last_regime or "pending",
    )


@router.delete("/session/{symbol}", summary="Remove a streaming session")
async def remove_session(symbol: str):
    await registry.remove(symbol)
    return {"status": "removed", "symbol": symbol}
