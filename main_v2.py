"""
Dynamic Regime Risk Management Platform — FastAPI Service v2.0

Batch: POST /regime/detect | /size/kelly | /risk/analyse | /analyse/full
Stream: POST /stream/seed | /stream/tick | WS /ws/{symbol} | GET /sse/{symbol} | GET /sse/alerts
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from regime_platform_v2.routers import regime, sizing, risk, pipeline
from regime_platform_v2.routers import stream_rest, stream_ws


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dynamic Regime Risk Management Platform",
        version="2.0.0",
        description="""
## v2.0 — Live Streaming Support

### Streaming workflow

```
1. POST /stream/seed   { "symbol": "SPY", "prices": [...500+ bars] }
2. WS   /ws/SPY        {"type":"tick","symbol":"SPY","price":512.34}
                       ← receives full regime+sizing+risk JSON per bar
3. GET  /sse/alerts    subscribe to global regime-change event broadcast
```

### Session model

Each symbol has a **StreamSession**:
- Rolling price deque (default 500 bars)  
- Background HMM re-fit every N bars (default 50)  
- Regime-change alerts pushed to all connected WS/SSE subscribers  
- Multiple concurrent clients per symbol  

### Transport options

| Transport | Endpoint | Best for |
|-----------|----------|---------|
| REST tick | POST /stream/tick | server-to-server pipelines |
| WebSocket | WS /ws/{symbol} | full-duplex, browser or service |
| SSE symbol | GET /sse/{symbol} | browser EventSource |
| SSE alerts | GET /sse/alerts | regime-change alert hub |
        """,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    app.include_router(regime.router)
    app.include_router(sizing.router)
    app.include_router(risk.router)
    app.include_router(pipeline.router)
    app.include_router(stream_rest.router)
    app.include_router(stream_ws.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse({
            "service": "Dynamic Regime Risk Management Platform",
            "version": "2.0.0",
            "docs": "/docs",
        })

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "version": "2.0.0"}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)
