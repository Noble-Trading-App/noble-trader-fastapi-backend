# Dynamic Regime Risk Management Platform

A production-grade FastAPI web service for quantitative trading risk management. The platform detects market regimes with Hidden Markov Models, dynamically gates position sizes via the Kelly Criterion, computes real-time risk metrics (VaR/CVaR), and streams live regime snapshots and alerts over WebSocket and SSE.

---

## Architecture

```
regime_platform/
├── core/
│   ├── regime_engine.py      # Dual GaussianHMM (vol + trend)
│   ├── position_sizer.py     # Fractional Kelly + vol-scaling + regime gating
│   └── risk_manager.py       # VaR, CVaR, drawdown, stop/TP
├── models/
│   ├── schemas.py            # Pydantic v2 batch request/response models
│   └── stream_schemas.py     # Pydantic v2 streaming models
├── routers/
│   ├── regime.py             # POST /regime/detect
│   ├── sizing.py             # POST /size/kelly
│   ├── risk.py               # POST /risk/analyse
│   ├── pipeline.py           # POST /analyse/full
│   ├── stream_rest.py        # POST /stream/seed | /stream/tick | /stream/ticks
│   └── stream_ws.py          # WS /ws/{symbol} | GET /sse/{symbol} | GET /sse/alerts
└── services/
    ├── stream_session.py     # Per-symbol stateful streaming engine
    └── registry.py           # Singleton session store
main_v1.py                       # v1 batch-only app factory
main.py                       # v2 app factory (batch + streaming)
requirements.txt
test_platform.py              # Batch pipeline smoke test
test_streaming.py             # Streaming layer smoke test
```

---

## Library Choices & Rationale

| Library | Role | Why |
|---------|------|-----|
| **FastAPI** | Web framework | Async, auto-docs, Pydantic v2 native, fastest Python HTTP |
| **hmmlearn** | Hidden Markov Models | Mature Gaussian HMM with Viterbi + posterior probabilities |
| **scipy** | Stats primitives | VaR/CVaR, normal distribution, optimisation scaffolding |
| **numpy** | Numerical core | Vectorised return/vol calculations, no pandas overhead in hot path |
| **pydantic v2** | Validation & serialisation | Rust-core validation, discriminated unions, field constraints |
| **uvicorn** | ASGI server | Production-ready ASGI, supports reload + multi-worker |

> **Why not pynamical/simupy?**
> Both are simulation/dynamical-systems libraries rather than statistical inference or risk engines. `pynamical` is a chaos/bifurcation visualiser; `simupy` is a block-diagram ODE solver. For a trading risk service, `hmmlearn` + `scipy` give the right statistical rigour. Either could be layered in as a `/simulate` extension endpoint for scenario generation or regime transition modelling.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run (v2 — batch + streaming)
uvicorn main:app --reload --port 8000

# Interactive docs
open http://localhost:8000/docs
```

---

## Endpoints

### Batch (one-shot historical analysis)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyse/full` | ⭐ Full pipeline: regime + sizing + risk in one request |
| `POST` | `/regime/detect` | HMM regime classification only |
| `POST` | `/size/kelly` | Regime-gated Kelly position sizing only |
| `POST` | `/risk/analyse` | VaR, CVaR, drawdown, stop/TP only |

### Live Streaming

| Method | Path | Best for |
|--------|------|---------|
| `POST` | `/stream/seed` | Seed a symbol session with historical prices |
| `POST` | `/stream/tick` | Push one price tick → receive regime snapshot |
| `POST` | `/stream/ticks` | Batch replay / backtest tick ingestion |
| `GET` | `/stream/sessions` | List all active sessions and their stats |
| `GET` | `/stream/session/{symbol}` | Single session status |
| `DELETE` | `/stream/session/{symbol}` | Remove a session |
| `WS` | `/ws/{symbol}` | Full-duplex WebSocket stream |
| `GET` | `/sse/{symbol}` | Server-Sent Events push stream (per symbol) |
| `GET` | `/sse/alerts` | SSE global regime-change alert broadcast |

---

## Batch Endpoint Details

### `POST /regime/detect`
Fits dual 3-state Gaussian HMMs on a price series and returns:
- Posterior probabilities for volatility regimes: `low / medium / high`
- Posterior probabilities for trend regimes: `bear / neutral / bull`
- Combined `regime_label` (e.g. `low_vol_bull`)
- `risk_multiplier` [0.25 – 1.50] for downstream sizing

### `POST /size/kelly`
Computes fractional Kelly position size with three layers of scaling:
1. **Fractional Kelly** — reduce full Kelly by a fraction (default 50%)
2. **Vol-scaling** — scale to hit a target annualised volatility
3. **Regime gating** — apply `risk_multiplier` from HMM

### `POST /risk/analyse`
Full risk decomposition:
- Historical VaR 95%/99% and CVaR (Expected Shortfall)
- Max drawdown, Sortino ratio, Calmar ratio
- Regime-adjusted max loss limit
- Suggested stop-loss (2× CVaR95) and take-profit (3:1 R/R)
- Risk budget utilisation

### `POST /analyse/full`
One-shot pipeline — prices in, full `{regime, sizing, risk}` JSON out.

---

## Streaming Workflow

### 1. Seed the session
```bash
curl -X POST http://localhost:8000/stream/seed \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "prices": [...200+ bars], "kelly_fraction": 0.5}'
```

### 2a. REST tick ingestion (server-to-server pipelines)
```bash
curl -X POST http://localhost:8000/stream/tick \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "price": 512.34}'
```

### 2b. WebSocket (full-duplex — algo engines, browser UIs)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/SPY');

// Seed
ws.send(JSON.stringify({ type: "seed", symbol: "SPY", prices: [...] }));

// Push ticks — receive regime+sizing+risk JSON on every bar
ws.send(JSON.stringify({ type: "tick", symbol: "SPY", price: 512.34 }));

ws.onmessage = e => {
  const tick = JSON.parse(e.data);
  console.log(tick.regime_label, tick.recommended_f, tick.var_95);
};

// Or subscribe to be pushed ticks ingested by other producers
ws.send(JSON.stringify({ type: "subscribe", symbol: "SPY" }));
```

### 2c. SSE (browser dashboards, read-only consumers)
```javascript
// Per-symbol stream
const feed = new EventSource('http://localhost:8000/sse/SPY');
feed.onmessage = e => console.log(JSON.parse(e.data));

// Global regime-change alerts across all symbols
const alerts = new EventSource('http://localhost:8000/sse/alerts');
alerts.onmessage = e => {
  const alert = JSON.parse(e.data);
  // { symbol, previous, current, severity: "info|warning|critical", message }
};
```

---

## Example Batch Request & Response

```bash
curl -X POST http://localhost:8000/analyse/full \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "prices": [100.0, 101.2, ...],
    "kelly_fraction": 0.5,
    "target_vol": 0.15,
    "base_risk_limit": 0.02
  }'
```

```json
{
  "symbol": "SPY",
  "regime": {
    "vol_state": "low",
    "trend_state": "bull",
    "regime_label": "low_vol_bull",
    "confidence": 0.8234,
    "risk_multiplier": 1.5
  },
  "sizing": {
    "full_kelly_f": 0.8421,
    "fractional_f": 0.4211,
    "vol_scaled_f": 0.3247,
    "regime_gated_f": 0.4870,
    "recommended_f": 0.4870,
    "sharpe_ratio": 1.34
  },
  "risk": {
    "var_95": 0.0187,
    "cvar_95": 0.0263,
    "max_drawdown": -0.1124,
    "sortino_ratio": 1.82,
    "suggested_stop": -0.0526,
    "suggested_tp": 0.0789
  }
}
```

---

## Example Streaming Tick Response

Every tick — whether via REST, WebSocket, or SSE — emits this structure:

```json
{
  "symbol": "SPY",
  "ts": 1710000512.0,
  "price": 512.34,
  "n_bars": 247,
  "regime_label": "low_vol_bull",
  "vol_state": "low",
  "trend_state": "bull",
  "vol_probs":   { "low": 0.82, "medium": 0.15, "high": 0.03 },
  "trend_probs": { "bear": 0.05, "neutral": 0.12, "bull": 0.83 },
  "confidence": 0.6806,
  "risk_multiplier": 1.5,
  "recommended_f": 0.4870,
  "sharpe_ratio": 1.34,
  "var_95": 0.0187,
  "cvar_95": 0.0263,
  "suggested_stop": -0.0526,
  "suggested_tp": 0.0789,
  "regime_changed": false,
  "alert": null,
  "refit_count": 4
}
```

---

## Regime → Risk Multiplier Map

| Vol \ Trend | Bear | Neutral | Bull |
|------------|------|---------|------|
| **Low**    | 0.75 | 1.28    | 1.50 |
| **Medium** | 0.50 | 0.85    | 1.00 |
| **High**   | 0.25 | 0.43    | 0.50 |

---

## StreamSession Internals

Each symbol maintains an independent `StreamSession` with three concurrent responsibilities:

**Tick path (~7ms/tick)** runs synchronously on every price update: predict regime posteriors → compute Kelly fraction → run VaR/CVaR → push snapshot to all subscriber queues.

**Background HMM refit** (every 50 bars by default) runs via `asyncio.run_in_executor` so the main event loop never stalls during the EM fitting step.

**Alert debounce** uses a 3-bar hysteresis window — a new regime must persist for 3 consecutive bars before a `RegimeAlert` fires. This prevents noisy micro-transitions from flooding downstream consumers with false alerts.

Regime-change alerts carry a `severity` field: `info` for benign transitions, `warning` for bear/high-vol entries, and `critical` for `high_vol_bear` regimes.

---

## Extending the Platform

- **Simulation** — add `simupy` for ODE-based regime transition simulation as a `/simulate` endpoint
- **Chaos analysis** — add `pynamical` for bifurcation/Lyapunov analysis of portfolio dynamics
- **Multi-asset** — the `SessionRegistry` already supports concurrent per-symbol sessions; add a `/portfolio` aggregation layer
- **GPU acceleration** — replace `hmmlearn` with `pomegranate` (PyTorch backend) for GPU-accelerated HMM on large tick streams
- **Persistence** — swap `collections.deque` price buffer for Redis Streams to survive process restarts
- **Authentication** — add FastAPI `Depends` JWT middleware to the WebSocket handshake

---

## Notes

- Minimum 51 price bars required to fit the HMM (50 returns + 1 for feature computation)
- HMM convergence warnings on short series are expected; increase `refit_every` or warm up with more seed bars for stability
- All position fractions are in `[0, 1]` portfolio-fraction space — multiply by notional capital externally
- VaR/CVaR are historical (empirical), not parametric — more robust for fat-tailed asset return distributions
- The `covariance_type` is set to `"diag"` (vs `"full"`) for numerical stability with rolling windows; switch to `"full"` if using 500+ seed bars
