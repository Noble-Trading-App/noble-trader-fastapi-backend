"""
Real-time streaming endpoints.

  WS  /ws/{symbol}              — full-duplex WebSocket stream
  GET /sse/{symbol}             — Server-Sent Events one-way stream
  GET /sse/alerts               — SSE stream for all regime-change alerts

WebSocket protocol (JSON messages):
  Client → Server:
    {"type": "seed",      "symbol": "SPY", "prices": [...]}
    {"type": "tick",      "symbol": "SPY", "price": 512.34, "ts": 1710000000.0}
    {"type": "subscribe", "symbol": "SPY"}
    {"type": "ping"}

  Server → Client:
    tick snapshots (TickResponse JSON)
    {"type": "pong"}
    {"type": "error", "detail": "..."}
    {"type": "alert", ...}
"""

from __future__ import annotations
import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse

from ..services.registry import registry
from ..core.regime_engine import RegimeHMM
from ..services.stream_session import RegimeAlert
from ..models.stream_schemas import WsMessage

router = APIRouter(tags=["Real-Time Streaming"])

# Global alert broadcast queue
_alert_broadcast: asyncio.Queue = asyncio.Queue(maxsize=1000)


# ─── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws/{symbol}")
async def websocket_stream(websocket: WebSocket, symbol: str):
    """
    Full-duplex WebSocket for a single symbol.

    Connect, then send JSON messages to seed / push ticks.
    The server pushes back a regime snapshot after every tick
    and an alert message on regime transitions.

    Example client flow:
      1. {"type":"seed",  "symbol":"SPY", "prices":[...]}
      2. {"type":"tick",  "symbol":"SPY", "price":512.34}
      3. ... receive tick snapshots ...
    """
    await websocket.accept()

    async def _send(data: dict):
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    # Register alert callback for this connection
    async def _on_alert(alert: RegimeAlert):
        await _send({
            "type":     "alert",
            "symbol":   alert.symbol,
            "ts":       alert.ts,
            "previous": alert.previous,
            "current":  alert.current,
            "message":  alert.message,
            "severity": alert.severity,
        })
        await _alert_broadcast.put(alert.__dict__)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = WsMessage.model_validate_json(raw)
            except Exception as e:
                await _send({"type": "error", "detail": f"Invalid message: {e}"})
                continue

            # ── ping ──────────────────────────────────────────────────────────
            if msg.type == "ping":
                await _send({"type": "pong", "ts": time.time()})

            # ── seed ──────────────────────────────────────────────────────────
            elif msg.type == "seed":
                if not msg.prices or len(msg.prices) < 51:
                    await _send({"type": "error", "detail": "seed requires ≥51 prices"})
                    continue
                session = await registry.get_or_create(symbol=msg.symbol or symbol)
                session.add_alert_callback(_on_alert)
                await session.seed(msg.prices)
                await _send({
                    "type":         "seeded",
                    "symbol":       session.symbol,
                    "ready":        session.is_ready,
                    "n_bars":       len(session.price_buffer),
                    "refit_count":  session._refit_count,
                })

            # ── tick ──────────────────────────────────────────────────────────
            elif msg.type == "tick":
                if msg.price is None:
                    await _send({"type": "error", "detail": "tick requires 'price'"})
                    continue
                session = await registry.get(msg.symbol or symbol)
                if session is None:
                    await _send({"type": "error", "detail": "session not seeded — send seed first"})
                    continue
                tick = await session.tick(msg.price, msg.ts or time.time())
                if tick is None:
                    await _send({"type": "buffering", "n_bars": len(session.price_buffer)})
                else:
                    d = tick.to_dict()
                    d["type"] = "tick"
                    d["vol_probs"]   = dict(zip(RegimeHMM.VOL_LABELS,   d["vol_probs"]))
                    d["trend_probs"] = dict(zip(RegimeHMM.TREND_LABELS,  d["trend_probs"]))
                    await _send(d)

            # ── subscribe (push mode — client just listens) ───────────────────
            elif msg.type == "subscribe":
                session = await registry.get(msg.symbol or symbol)
                if session is None:
                    await _send({"type": "error", "detail": "no session for symbol"})
                    continue
                q = session.subscribe()
                session.add_alert_callback(_on_alert)
                await _send({"type": "subscribed", "symbol": session.symbol})
                try:
                    while True:
                        try:
                            tick = await asyncio.wait_for(q.get(), timeout=30)
                            d = tick.to_dict()
                            d["type"] = "tick"
                            d["vol_probs"]   = dict(zip(RegimeHMM.VOL_LABELS,  d["vol_probs"]))
                            d["trend_probs"] = dict(zip(RegimeHMM.TREND_LABELS, d["trend_probs"]))
                            await _send(d)
                        except asyncio.TimeoutError:
                            await _send({"type": "heartbeat", "ts": time.time()})
                finally:
                    session.unsubscribe(q)

            else:
                await _send({"type": "error", "detail": f"Unknown type '{msg.type}'"})

    except WebSocketDisconnect:
        pass


# ─── SSE — per symbol ────────────────────────────────────────────────────────

@router.get(
    "/sse/{symbol}",
    summary="Server-Sent Events stream for a symbol",
    response_class=StreamingResponse,
)
async def sse_symbol(symbol: str):
    """
    SSE stream that pushes a JSON regime snapshot on every tick received
    by the server for `symbol`. Subscribe from any browser with EventSource.

    Example JS:
        const es = new EventSource('/sse/SPY');
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    session = await registry.get(symbol)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session for '{symbol}'. Seed first.")

    q = session.subscribe()

    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'connected', 'symbol': symbol})}\n\n"
        try:
            while True:
                try:
                    tick = await asyncio.wait_for(q.get(), timeout=15)
                    d = tick.to_dict()
                    d["vol_probs"]   = dict(zip(RegimeHMM.VOL_LABELS,   d["vol_probs"]))
                    d["trend_probs"] = dict(zip(RegimeHMM.TREND_LABELS,  d["trend_probs"]))
                    yield f"data: {json.dumps(d)}\n\n"
                except asyncio.TimeoutError:
                    yield f": heartbeat {time.time()}\n\n"
        except asyncio.CancelledError:
            session.unsubscribe(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── SSE — global alerts ─────────────────────────────────────────────────────

@router.get(
    "/sse/alerts",
    summary="Global SSE stream for all regime-change alerts across all symbols",
    response_class=StreamingResponse,
)
async def sse_alerts():
    """
    Broadcasts regime-change alerts for all symbols to connected listeners.
    Only fires on regime state transitions (not every tick).

    Severity levels: info | warning | critical
    """
    # Each consumer gets its own queue fed from the broadcast
    local_q: asyncio.Queue = asyncio.Queue(maxsize=200)

    # Fan-out task
    async def fan_out():
        while True:
            alert_dict = await _alert_broadcast.get()
            try:
                local_q.put_nowait(alert_dict)
            except asyncio.QueueFull:
                pass

    task = asyncio.create_task(fan_out())

    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'type': 'connected', 'scope': 'alerts'})}\n\n"
        try:
            while True:
                try:
                    alert = await asyncio.wait_for(local_q.get(), timeout=20)
                    alert["type"] = "alert"
                    yield f"data: {json.dumps(alert)}\n\n"
                except asyncio.TimeoutError:
                    yield f": heartbeat {time.time()}\n\n"
        except asyncio.CancelledError:
            task.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
