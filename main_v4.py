"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     Dynamic Regime Risk Management Platform — FastAPI Service  v3.0.0       ║
║                                                                              ║
║  v3.0 additions (roadmap items):                                             ║
║    GPU HMM       GET  /gpu/capabilities — pomegranate backend auto-select    ║
║                  POST /gpu/benchmark    — fit/predict latency benchmark      ║
║    Feed adapters POST /feeds/start      — Alpaca / Binance / IB live feeds   ║
║                  POST /feeds/stop       — stop all feed adapters             ║
║                  GET  /feeds/status     — bar counts and adapter health      ║
║    Correlation   POST /correlation/detect — DCC-based portfolio regime       ║
║    Optimisation  POST /optimise           — drawdown-controlled max-Sharpe   ║
║                  POST /optimise/full      — correlation + optimise one-shot  ║
║                                                                              ║
║  All v2.x endpoints unchanged.                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Environment variables
─────────────────────
  (All v2.x vars still apply)
  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL  — Alpaca feed
  BINANCE_TESTNET=true                                — Binance testnet
  IB_HOST, IB_PORT, IB_CLIENT_ID                     — Interactive Brokers
"""

from contextlib import asynccontextmanager
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# v2.x routers
from regime_platform.routers import regime, sizing, risk, pipeline
from regime_platform.routers import stream_rest, stream_ws
from regime_platform.routers import simulate, portfolio, auth_router

# v3.0 routers
from regime_platform.routers import gpu, feeds, multi_asset

from regime_platform.services.redis_persistence import persistence
from regime_platform.core.gpu_engine import gpu_capabilities

log = logging.getLogger("regime.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up v3.0.0 ...")
    log.info(f"GPU: {gpu_capabilities()}")

    if persistence.enabled:
        symbols = await persistence.list_persisted_symbols()
        log.info(f"Redis: restoring {len(symbols)} symbol(s)")
        from regime_platform.services.registry import registry
        for symbol in symbols:
            meta    = await persistence.load_meta(symbol)
            session = await registry.get_or_create(symbol=symbol, **meta)
            n       = await persistence.restore(session)
            if n >= 81:
                import asyncio
                await asyncio.get_event_loop().run_in_executor(None, session._refit)
                log.info(f"  ✓ {symbol}: restored {n} bars")

    yield

    await persistence.close()
    from regime_platform.routers.feeds import _manager
    if _manager:
        await _manager.stop()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dynamic Regime Risk Management Platform",
        version="3.0.0",
        lifespan=lifespan,
        description="""
## v3.0.0 — GPU HMM · Live Feeds · Correlation Regimes · Portfolio Optimisation

### New in v3.0

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **GPU HMM** | `GET /gpu/capabilities` | Check GPU + backend |
| **GPU HMM** | `POST /gpu/benchmark` | Latency benchmark |
| **Live feeds** | `POST /feeds/start` | Start Alpaca/Binance/IB adapters |
| **Live feeds** | `GET /feeds/status` | Adapter health + bar counts |
| **Correlation** | `POST /correlation/detect` | DCC correlation regime |
| **Optimise** | `POST /optimise` | Drawdown-controlled max-Sharpe |
| **Optimise** | `POST /optimise/full` | Correlation + optimise one-shot |

### GPU HMM

When `pomegranate ≥ 1.0` is installed and a CUDA/MPS device is available,
all HMM fit/predict calls run on GPU (typically 10–50× faster on large bars).
Transparent fallback to hmmlearn on CPU — no code changes needed.

```bash
pip install pomegranate torch
GET /gpu/capabilities   # → {"active_device": "cuda", "hmm_backend": "GpuRegimeHMM"}
```

### Live feed workflow

```bash
# 1. Seed symbols
POST /stream/seed  {"symbol": "SPY", "prices": [...200+ bars...]}

# 2. Start live feed (bars auto-routed into seeded sessions)
POST /feeds/start
[{"source": "alpaca", "symbols": ["SPY","QQQ"], "bar_size": "1Min"}]

# 3. Stream via WebSocket or SSE (same as before)
WS /ws/SPY
GET /sse/alerts
```

### Portfolio optimisation workflow

```bash
POST /optimise/full
{
  "symbols": ["SPY", "QQQ", "GLD", "TLT"],
  "returns_matrix": [[r_spy, r_qqq, r_gld, r_tlt], ...],  # n_bars × 4
  "max_dd_limit": 0.15,
  "max_weight": 0.40
}
# → correlation regime + drawdown-controlled optimal weights
```
        """,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # v2.x routers
    for r in [regime, sizing, risk, pipeline, stream_rest, stream_ws,
              simulate, portfolio, auth_router]:
        app.include_router(r.router)

    # v3.0 routers
    app.include_router(gpu.router)
    app.include_router(feeds.router)
    app.include_router(multi_asset.router)

    @app.get("/", include_in_schema=False)
    async def root():
        caps = gpu_capabilities()
        return JSONResponse({
            "service": "Dynamic Regime Risk Management Platform",
            "version": "3.0.0",
            "docs":    "/docs",
            "gpu":     caps,
        })

    @app.get("/health", tags=["Health"])
    async def health():
        from regime_platform.services.registry import registry
        symbols = await registry.list_symbols()
        caps    = gpu_capabilities()
        return {
            "status":      "ok",
            "version":     "3.0.0",
            "sessions":    len(symbols),
            "redis":       persistence.enabled,
            "gpu_backend": caps["hmm_backend"],
            "device":      caps["active_device"],
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main_v4:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
