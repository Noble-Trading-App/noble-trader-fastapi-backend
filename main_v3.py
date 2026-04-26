"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Dynamic Regime Risk Management Platform — FastAPI Service  v2.1.0       ║
║                                                                              ║
║  v2.1 additions (roadmap items):                                             ║
║    POST /simulate/{symbol}  — Markov-chain regime transition simulation      ║
║    GET  /portfolio          — Multi-symbol aggregated regime + risk          ║
║    Redis Streams            — Optional price buffer persistence              ║
║    JWT / API-Key auth       — WebSocket + HTTP endpoint protection           ║
║    POST /auth/token         — Issue JWT access tokens                        ║
║    GET  /auth/me            — Current user info                              ║
║                                                                              ║
║  All v2.0 endpoints unchanged:                                               ║
║    POST /analyse/full | /regime/detect | /size/kelly | /risk/analyse         ║
║    POST /stream/seed | /stream/tick | WS /ws/{symbol}                        ║
║    GET  /sse/{symbol} | /sse/alerts | /stream/sessions                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Environment variables
─────────────────────
  JWT_SECRET_KEY   signing secret for JWT (required for auth)
  JWT_EXPIRE_MINS  token lifetime (default 60)
  API_KEYS         comma-separated API keys (optional)
  AUTH_ENABLED     set "false" to disable auth in development
  AUTH_USERS       "user:pass:role" entries for /auth/token endpoint
  REDIS_URL        Redis connection string (optional, enables persistence)
"""

from contextlib import asynccontextmanager
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from regime_platform.routers import regime, sizing, risk, pipeline
from regime_platform.routers import stream_rest, stream_ws
from regime_platform.routers import simulate, portfolio, auth_router
from regime_platform.services.redis_persistence import persistence

log = logging.getLogger("regime.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info("Starting up v2.1.0 ...")

    # Restore any persisted sessions from Redis on startup
    if persistence.enabled:
        symbols = await persistence.list_persisted_symbols()
        log.info(f"Redis persistence active — found {len(symbols)} persisted symbol(s): {symbols}")
        from regime_platform.services.registry import registry
        for symbol in symbols:
            meta    = await persistence.load_meta(symbol)
            session = await registry.get_or_create(symbol=symbol, **meta)
            n       = await persistence.restore(session)
            if n >= 81:
                import asyncio
                await asyncio.get_event_loop().run_in_executor(None, session._refit)
                log.info(f"  ✓ {symbol}: restored {n} bars, HMM re-fitted")
            else:
                log.info(f"  ⚠ {symbol}: restored {n} bars (< 81 min, pending seed)")
    else:
        log.info("Redis not configured — starting with empty in-memory sessions")

    yield  # ── app is running ──

    # Graceful shutdown
    await persistence.close()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dynamic Regime Risk Management Platform",
        version="2.1.0",
        lifespan=lifespan,
        description="""
## v2.1.0 — Simulation · Portfolio · Redis Persistence · JWT Auth

### New in v2.1

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **Simulation** | `POST /simulate/{symbol}` | Monte Carlo regime transition paths |
| **Portfolio** | `GET /portfolio` | Multi-symbol aggregated risk view |
| **Auth** | `POST /auth/token` | Issue JWT access tokens |
| **Auth** | `GET /auth/me` | Verify token and role |
| **Persistence** | (background) | Redis Streams price buffer survival |

### Authentication

Set `AUTH_ENABLED=false` in development to bypass auth.

In production, configure:
```
JWT_SECRET_KEY=your-secret
AUTH_USERS=admin:pass:admin,trader:pass123:trader
API_KEYS=svc-key-1,svc-key-2
```

Pass `Authorization: Bearer <token>` on protected endpoints,
or `?token=<jwt>` on WebSocket connections.

### Simulation workflow

```
POST /simulate/SPY
{
  "prices": [...200 bars...],
  "horizon": 20,
  "n_paths": 500
}
→ price fan (p5/p25/median/p75/p95), expected risk multiplier per step,
  terminal VaR/CVaR, most likely terminal regime
```

### Portfolio workflow

```
# 1. Seed multiple symbols
POST /stream/seed  {"symbol": "SPY", "prices": [...]}
POST /stream/seed  {"symbol": "QQQ", "prices": [...]}
POST /stream/seed  {"symbol": "GLD", "prices": [...]}

# 2. Stream ticks to each
POST /stream/tick  {"symbol": "SPY", "price": 512.34}

# 3. Get portfolio view
GET /portfolio
→ per-symbol regime + Kelly f* + VaR, plus portfolio-level risk flags
```

### Redis Persistence

Set `REDIS_URL=redis://localhost:6379` to enable.
Price buffers survive process restarts. Sessions are restored on startup.
Without Redis, all state is in-memory (default, no config needed).
        """,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── v2.0 routers (unchanged) ──────────────────────────────────────────────
    app.include_router(regime.router)
    app.include_router(sizing.router)
    app.include_router(risk.router)
    app.include_router(pipeline.router)
    app.include_router(stream_rest.router)
    app.include_router(stream_ws.router)

    # ── v2.1 routers ─────────────────────────────────────────────────────────
    app.include_router(simulate.router)
    app.include_router(portfolio.router)
    app.include_router(auth_router.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({
            "service": "Dynamic Regime Risk Management Platform",
            "version": "2.1.0",
            "docs": "/docs",
            "new_in_v2.1": {
                "simulation": "POST /simulate/{symbol}",
                "portfolio":  "GET /portfolio",
                "auth":       "POST /auth/token",
                "persistence": "Redis Streams (set REDIS_URL env var)",
            },
        })

    @app.get("/health", tags=["Health"])
    async def health():
        from regime_platform.services.redis_persistence import persistence
        from regime_platform.services.registry import registry
        symbols = await registry.list_symbols()
        return {
            "status":   "ok",
            "version":  "2.1.0",
            "sessions": len(symbols),
            "redis":    persistence.enabled,
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main_v3:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
