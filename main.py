"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Dynamic Regime Risk Management Platform — FastAPI Service           ║
║                                                                              ║
║  Stack:  FastAPI · hmmlearn · scipy · numpy · pandas · pydantic v2          ║
║                                                                              ║
║  Endpoints:                                                                  ║
║    POST /regime/detect      — HMM-based vol + trend regime classification    ║
║    POST /size/kelly         — Regime-gated fractional Kelly position sizing  ║
║    POST /risk/analyse       — VaR, CVaR, drawdown, stop/TP suggestions       ║
║    POST /analyse/full       — One-shot: all three in one request             ║
║    POST /simulate/{symbol}  — Markov-chain regime transition simulation      ║
║    GET /portfolio           — Multi-symbol aggregated regime + risk          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from regime_platform.routers import pipeline, portfolio, regime, risk, simulate, sizing

# ─── App Factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="Noble Trader Dynamic Regime Risk Management Platform",
        description="""
## Overview

A quantitative trading risk management web service built on:

| Component | Library | Purpose |
|-----------|---------|---------|
| **Regime Detection** | `hmmlearn` GaussianHMM | Classify market into volatility × trend regimes |
| **Position Sizing** | `scipy` + Kelly formula | Fractional Kelly with vol-scaling & regime gating |
| **Risk Analysis** | `numpy` / `scipy.stats` | VaR, CVaR, drawdown, Sortino, Calmar |
| **API Framework** | `FastAPI` + `pydantic v2` | Typed, validated, auto-documented REST API |

## Regime Labels

The HMM produces a **2D regime** combining:
- **Volatility**: `low` · `medium` · `high`
- **Trend**: `bear` · `neutral` · `bull`

Combined label example: `low_vol_bull`, `high_vol_bear`, etc.

## Risk Multiplier

Each regime maps to a **risk_multiplier** [0.25 – 1.50] that scales the Kelly fraction:
- `low_vol_bull` → 1.5× (max risk-on)
- `high_vol_bear` → 0.25× (max defensive)

## Recommended Workflow

```
POST /analyse/full  →  regime + sizing + risk in one call
```

Or use individual endpoints for targeted analysis.
        """,
        version="1.0.0",
        contact={
            "name": "Noble Trader Dynamic Regime Risk Platform",
            "email": "nobletrader@0xdweb.com",
        },
        license_info={"name": "MIT"},
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(regime.router)
    app.include_router(sizing.router)
    app.include_router(risk.router)
    app.include_router(pipeline.router)
    app.include_router(simulate.router)
    app.include_router(portfolio.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            {
                "service": "Dynamic Regime Risk Management Platform",
                "version": "1.0.0",
                "docs": "/docs",
                "endpoints": {
                    "regime_detect": "POST /regime/detect",
                    "kelly_sizing": "POST /size/kelly",
                    "risk_analysis": "POST /risk/analyse",
                    "full_pipeline": "POST /analyse/full",
                    "simulate": "POST /simulate/{symbol}",
                    "portfolio": "GET /portfolio",
                },
            }
        )

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
