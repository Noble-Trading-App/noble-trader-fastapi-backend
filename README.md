# Dynamic Regime Risk Management Platform

A production-grade FastAPI web service for quantitative trading risk management.
Detects market regimes with 4-state Hidden Markov Models, gates position sizes
via a unified Dynamic Masaniello + Kelly Criterion engine, computes real-time risk
metrics (VaR/CVaR), streams live regime snapshots and alerts over WebSocket and SSE,
and produces 24-feature observation vectors for downstream RL/ML policy layers.

---

## Version

**v3.1.0** — GPU HMM · Live feeds · Correlation regimes · Portfolio optimisation ·
Dynamic Masaniello+Kelly sizer · Streaming test tools

---

## Project Structure

```
regime_risk_platform/
│
├── regime_platform/                  Python package
│   ├── core/
│   │   ├── regime_engine.py          4-state dual GaussianHMM + _sanitize_model
│   │   ├── obs_builder.py            24-feature InferenceObservationBuilder
│   │   ├── position_sizer.py         Unified Kelly (Path A) + Masaniello (Path B)
│   │   ├── risk_manager.py           VaR, CVaR, drawdown, stop/TP
│   │   ├── simulator.py              Markov-chain Monte Carlo regime simulation
│   │   ├── gpu_engine.py             Pomegranate GPU HMM + create_regime_hmm()
│   │   ├── correlation_regime.py     DCC multi-asset correlation regime detector
│   │   └── portfolio_optimiser.py    Drawdown-controlled max-Sharpe optimiser
│   ├── adapters/
│   │   └── feed_adapters.py          Alpaca / Binance / IB live OHLCV adapters
│   ├── auth/
│   │   ├── jwt_auth.py               JWT + API-key auth, role-based access
│   │   └── clerk_auth.py             Clerk JWT verification, JWKS, webhooks
│   ├── models/
│   │   ├── schemas.py                Pydantic v2 batch I/O models
│   │   └── stream_schemas.py         Pydantic v2 streaming models
│   ├── routers/
│   │   ├── regime.py                 POST /regime/detect
│   │   ├── sizing.py                 POST /size/kelly
│   │   ├── risk.py                   POST /risk/analyse
│   │   ├── pipeline.py               POST /analyse/full
│   │   ├── stream_rest.py            POST /stream/seed | /tick | /ticks
│   │   ├── stream_.py              WS /ws/{symbol} | GET /sse/{symbol|alerts}
│   │   ├── simulate.py               POST /simulate/{symbol}
│   │   ├── portfolio.py              GET /portfolio
│   │   ├── auth_router.py            POST /auth/token | GET /auth/me | Clerk endpoints
│   │   ├── gpu.py                    GET /gpu/capabilities | POST /gpu/benchmark
│   │   ├── feeds.py                  POST /feeds/start | /stop | GET /feeds/status
│   │   └── multi_asset.py            POST /correlation/detect | /optimise | /optimise/full
│   └── services/
│       ├── stream_session.py         Per-symbol stateful streaming engine
│       ├── registry.py               Singleton session store
│       ├── portfolio_service.py      Multi-symbol regime + risk aggregation
│       └── redis_persistence.py      Optional Redis Streams price buffer persistence
│
├── main.py                           v1 — batch API only
├── main_v2.py                        v2 — batch + streaming
├── main_v3.py                        v2.1 — + simulation, portfolio, auth, Redis
├── main_v4.py                        v3.1 — all features (recommended entry point)
├── requirements.txt
│
├── docs/
│   ├── index.html                    Full documentation site (26 pages)
│   ├── ws-client.html                Interactive WebSocket test client
│   ├── openapi.yaml                  OpenAPI 3.1 specification
│   ├── CHANGELOG.md                  Complete version history (v1.0 → v3.1)
│   └── deployment.md                 Production operations guide
│
├── tests/
│   ├── test_platform.py              Smoke test: HMM + Kelly + risk
│   ├── test_platform_v4.py           4-state HMM: 16-cell multiplier table
│   ├── test_position_sizer.py        95-assertion Masaniello+Kelly suite
│   ├── test_obs_builder.py           24-feature observation builder (7 tests)
│   ├── test_streaming.py             StreamSession: asyncio ticks + alerts
│   ├── test_v21.py                   Simulation, portfolio, Redis, JWT
│   └── test_v30.py                   GPU, feeds, correlation, optimiser
│
└── tools/
    ├── curl_test.sh                  bash: health + full-pipeline + edge cases
    ├── stream_spy_ticks.py           Python: seed + 50 live ticks via REST
    ├── sse_alert_monitor.html        Browser: EventSource regime-change monitor
    └── payload_spy_200bars.json      Ready-made 200-bar SPY payload for curl
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start (recommended — all v3.1 features)
uvicorn main:app --port 8000 --reload

# 3. Swagger UI
open http://localhost:8000/docs

# 4. Local docs site
open docs/index.html
```

---

## Library Stack

| Library | Version | Role |
|---------|---------|------|
| **FastAPI** | ≥0.111 | Async web framework, auto-docs |
| **hmmlearn** | ≥0.3.2 | 4-state Gaussian HMM (CPU) |
| **scipy** | ≥1.13 | VaR/CVaR, SLSQP optimiser |
| **numpy** | ≥1.26 | Vectorised return/vol calculations |
| **pydantic v2** | ≥2.7 | Request/response validation |
| **uvicorn** | ≥0.29 | ASGI server |
| **python-jose** | ≥3.3 | JWT signing/verification |
| **redis** | ≥5.0 | Optional price buffer persistence |
| **pomegranate** | ≥1.0 *(optional)* | GPU-accelerated HMM (PyTorch) |
| **clerk-backend-api** | ≥0.1.0 | Clerk JWT authentication |
| **httpx** | ≥0.28.1 | Clerk JWKS fetching |
| **PyJWT** | ≥2.9.0,<3.0.0 | JWT token handling |
| **alpaca-py** | *(optional)* | Alpaca live feed adapter |
| **websockets** | *(optional)* | Binance live feed adapter |
| **ib_async** | *(optional)* | Interactive Brokers adapter |

---

## API Endpoints

### Batch (stateless, historical)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyse/full` | ⭐ Regime + sizing + risk in one request |
| `POST` | `/regime/detect` | HMM regime classification |
| `POST` | `/size/kelly` | Regime-gated Kelly position sizing |
| `POST` | `/risk/analyse` | VaR, CVaR, drawdown, stop/TP |
| `POST` | `/simulate/{symbol}` | Monte Carlo regime transition simulation |

### Multi-Asset

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/correlation/detect` | DCC portfolio correlation regime |
| `POST` | `/optimise` | Drawdown-controlled max-Sharpe |
| `POST` | `/optimise/full` | Correlation + optimise one-shot |
| `GET` | `/portfolio` | Multi-symbol aggregated risk view |

### Live Streaming

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/stream/seed` | Seed a symbol session |
| `POST` | `/stream/tick` | Push one price tick |
| `POST` | `/stream/ticks` | Batch tick replay |
| `GET` | `/stream/sessions` | List active sessions |
| `WS` | `/ws/{symbol}` | Full-duplex WebSocket |
| `GET` | `/sse/{symbol}` | SSE per-symbol stream |
| `GET` | `/sse/alerts` | SSE global regime-change alerts |

### Infrastructure

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/gpu/capabilities` | GPU hardware + HMM backend |
| `POST` | `/gpu/benchmark` | Fit/predict latency benchmark |
| `POST` | `/feeds/start` | Start Alpaca/Binance/IB live feed |
| `GET` | `/feeds/status` | Adapter health + bar counts |
| `POST` | `/auth/token` | Issue JWT access token |
| `GET` | `/auth/me` | Current user info |
| `GET` | `/auth/clerk/config` | Clerk auth configuration |
| `GET` | `/auth/clerk/me` | Clerk user info from JWT |
| `POST` | `/auth/clerk/verify` | Verify Clerk JWT token |
| `GET` | `/health` | Service health check |

---

## Regime Detection — 4-State HMM

Two independent 4-state Gaussian HMMs classify every price bar.
States are auto-labelled by sorting component means — no manual mapping.

| Dimension | States | Features |
|-----------|--------|---------|
| Volatility | `low` · `med_low` · `med_high` · `high` | 20-bar RV, 5-bar RV, \|return\| |
| Trend | `strong_bear` · `bear` · `bull` · `strong_bull` | return, 10-bar cum, 30-bar cum |

### Risk Multiplier Table — 4×4 = 16 regimes · range [0.10, 1.75]

| Vol \ Trend   | Strong Bear | Bear  | Bull  | Strong Bull |
|--------------|-------------|-------|-------|-------------|
| **Low**      | 0.70×       | 1.05× | 1.54× | **1.75×**   |
| **Med-Low**  | 0.48×       | 0.72× | 1.06× | 1.20×       |
| **Med-High** | 0.28×       | 0.42× | 0.62× | 0.70×       |
| **High**     | **0.14×**   | 0.21× | 0.31× | 0.35×       |

---

## Position Sizing — Dynamic Masaniello + Kelly

`core/position_sizer.py` provides two independent sizing paths:

### Path A — Portfolio-fraction Kelly (stateless)
Input: historical returns + `RegimeSnapshot`. Output: `recommended_f` ∈ [0, 1].

```python
from regime_platform.core.position_sizer import PositionSizer, PositionRequest

result = PositionSizer().size(PositionRequest(
    returns=log_returns, kelly_fraction=0.5, target_vol=0.15, regime=snap
))
print(result.recommended_f)
```

### Path B — Dynamic Masaniello (stateful, per-trade)

Formula: **`f_i = β × (0.5 + M_i) × Q_i × DD_i × V_i`**

| Factor | Formula | Clamp | Purpose |
|--------|---------|-------|---------|
| β | `base_risk` (default 0.50%) | — | Risk anchor |
| M_i | `(W − w) / (N − i + 1)` | [0.0, 1.5] | Batch urgency |
| Q_i | `prob_factor × regime_factor × conf_factor` | product | Quality gate |
| DD_i | `clip(1 − dd/max_dd, 0.25, 1.0)` | [0.25, 1.0] | Drawdown protection |
| V_i | `clip(ATR_base/ATR_current, 0.5, 1.5)` | [0.5, 1.5] | Volatility adjustment |

```python
from regime_platform.core.position_sizer import DynamicMasanielloSizer, SizingConfig

sizer = DynamicMasanielloSizer(SizingConfig(base_risk=0.005, batch_size=5, target_wins=3))
sizer.batch.peak_equity = 100_000

result = sizer.size_from_snapshot(
    snap=snap, equity=100_000,
    stop_distance_price=2.50, point_value=50.0,
    p_win=0.62, reward_risk=2.5,
    atr_baseline=12.0, atr_current=10.0,
)
print(result.summary())
# [✓ ALLOWED]  f=0.550%  $550  units=5.50  contracts=5  ...

sizer.batch.record(result, outcome=True, pnl=625.0)
sizer.batch.reset(equity=100_625)
```

---

## 24-Feature Observation Vector

`core/obs_builder.py` — `InferenceObservationBuilder`

**Critical rule:** always use the stateful class. Never use a standalone function.
If Markov features 14–19 show uniform values (~0.33), the uniform-prior bug is active.

| Indices | Features | Source |
|---------|---------|--------|
| 0–2 | log returns (1, 3, 10 bar) | Price series |
| 3–5 | normalised ATR, rolling vol, EMA distance | OHLCV |
| 6–9 | HHLL score, vol percentile, ATR ratio, vol slope | Derived |
| 10–13 | Raw HMM state posteriors (unsorted) | HMM |
| **14–19** | **Markov features: vol_prob_{low,med,high}, trend_prob_{down,neutral,up}** | **HMM** |
| 20–21 | Regime quality, state confidence | HMM |
| 22 | Masaniello pressure f×(1−f) | PositionSizer |
| 23 | Drawdown factor from peak | BatchState |

---

## Streaming Workflow

```bash
# 1. Seed
POST /stream/seed  {"symbol": "SPY", "prices": [...200+ bars]}

# 2a. REST ticks
POST /stream/tick  {"symbol": "SPY", "price": 512.34}

# 2b. WebSocket
ws://localhost:8000/ws/SPY
→ {"type":"seed","symbol":"SPY","prices":[...]}
→ {"type":"tick","symbol":"SPY","price":512.34}

# 2c. Subscribe (read-only)
GET /sse/SPY          # per-symbol tick stream
GET /sse/alerts       # global regime-change alerts

# 3. Portfolio view
GET /portfolio
```

---

## Testing

Run all suites in order — each layer builds on the previous:

```bash
# Unit (no server required)
python test_platform.py           # 3 checks  — smoke: HMM, Kelly, risk
python test_platform_v4.py        # 5 checks  — 4-state HMM, 16-cell table
python test_position_sizer.py     # 95 checks — Masaniello factors, gates, Kelly
python test_obs_builder.py        # 7 checks  — 24-feature builder, Markov guard

# Integration (no server required)
python test_streaming.py          # asyncio: 160 ticks, alerts, subscriber queue
export JWT_SECRET_KEY=test-secret
python test_v21.py                # simulation, portfolio, Redis no-op, JWT
python test_v30.py                # GPU, feed adapters, correlation, optimiser

# Live API (server must be running)
uvicorn main:app --port 8000 --reload
chmod +x curl_test.sh && ./curl_test.sh     # 6 curl assertions
python stream_spy_ticks.py                  # seed + 50 live ticks with formatted output
open sse_alert_monitor.html                 # browser SSE monitor for /sse/alerts
```

---

## Correlation Regime (Multi-Asset)

```bash
POST /optimise/full
{
  "symbols": ["SPY", "QQQ", "GLD", "TLT"],
  "returns_matrix": [[r_spy, r_qqq, r_gld, r_tlt], ...],
  "max_dd_limit": 0.15,
  "max_weight": 0.40
}
# Returns: correlation regime + drawdown-controlled optimal weights
```

| Correlation Regime | Mean \|ρ\| | Multiplier | Exposure |
|-------------------|-----------|-----------|---------|
| `low_corr` | < 0.20 | 1.00× | 100% |
| `mid_corr` | 0.20–0.50 | 0.85× | 90% |
| `high_corr` | 0.50–0.75 | 0.60× | 70% |
| `crisis` | > 0.75 | 0.35× | 50% |

---

## Authentication

### JWT Authentication (Built-in)

```bash
# Configure
export JWT_SECRET_KEY="your-signing-secret"
export AUTH_USERS="admin:pass:admin,trader:pass123:trader"
export API_KEYS="svc-key-1"
export AUTH_ENABLED=false   # dev mode — bypasses all auth

# Issue token
curl -X POST http://localhost:8000/auth/token \
  -d "username=trader&password=pass123"

# Use token
curl -H "Authorization: Bearer <jwt>" http://localhost:8000/portfolio

# WebSocket with token
ws://localhost:8000/ws/SPY?token=<jwt>
```

Roles: `admin` (full) · `trader` (read + write) · `viewer` (read-only)

### Clerk Authentication (New)

Clerk JWT authentication is now fully integrated. Configure Clerk keys in `.env.local`:

```bash
# .env.local
CLERK_PUBLISHABLE_KEY=pk_test_your_publishable_key
CLERK_SECRET_KEY=sk_test_your_secret_key
```

#### Clerk Endpoints

```bash
# Get Clerk configuration
curl https://your-domain.com/auth/clerk/config

# Verify a Clerk JWT token
curl -X POST https://your-domain.com/auth/clerk/verify \
  -H "Content-Type: application/json" \
  -d '{"token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..."}'

# Get current Clerk user info (requires valid Clerk JWT)
curl -H "Authorization: Bearer <clerk_jwt>" \
  https://your-domain.com/auth/clerk/me
```

#### Protecting Endpoints with Clerk

```python
from regime_platform.auth import get_current_clerk_user, require_clerk_admin

@router.get("/protected")
async def protected_route(user = Depends(get_current_clerk_user)):
    return {"user_id": user.sub, "email": user.email}

@router.post("/admin-only")
async def admin_only(user = Depends(require_clerk_admin)):
    return {"message": "Admin access granted"}
```

#### Clerk Features

- ✅ **JWT Verification**: Verify Clerk-issued JWT tokens using Clerk's JWKS
- ✅ **JWKS Fetching**: Automatic key rotation support
- ✅ **Role-based Access**: Admin/trader/viewer roles from token claims
- ✅ **Webhook Verification**: Secure Clerk webhook signature validation
- ✅ **FastAPI Integration**: Ready-to-use dependency injectors
- ✅ **Configuration Endpoints**: `/auth/clerk/config` for client setup

#### Clerk Token Data

Verified Clerk tokens provide rich user data:
- `sub`: Clerk user ID
- `email`: User email address
- `first_name`, `last_name`: User's name
- `username`: User's username
- `role`: User role (admin/trader/viewer)
- Full token claims available via `user.claims`

#### Webhook Security

```python
from regime_platform.auth import verify_clerk_webhook

@router.post("/webhooks/clerk")
async def clerk_webhook(request: Request):
    payload = await verify_clerk_webhook(request)
    # Process verified webhook payload
    return {"status": "processed"}
```

Clerk authentication works alongside the existing JWT system, allowing flexible migration paths.

### Dual Authentication Support

Both `/simulate` and `/portfolio` endpoints now support both authentication systems:

```python
# Endpoints accept both JWT and Clerk users
from regime_platform.auth import TokenData, ClerkTokenData
from typing import Union

@router.post("/simulate/{symbol}")
async def simulate_symbol(
    symbol: str,
    req: SimulateRequest,
    user: Union[TokenData, ClerkTokenData] = Depends(get_current_user),
):
    # Works with both JWT and Clerk authentication
    pass
```

**Usage examples:**

```bash
# Using JWT token
curl -H "Authorization: Bearer <jwt_token>" \
  https://your-domain.com/simulate/SPY

# Using Clerk JWT token
curl -H "Authorization: Bearer <clerk_jwt_token>" \
  https://your-domain.com/simulate/SPY

# Using API key (service-to-service)
curl -H "X-API-Key: your-api-key" \
  https://your-domain.com/portfolio
```

All endpoints automatically support:
- ✅ **JWT tokens** (built-in system)
- ✅ **Clerk JWT tokens** (new Clerk integration)
- ✅ **API keys** (service-to-service)
- ✅ **Development mode** (AUTH_ENABLED=false)

This provides maximum flexibility for different authentication scenarios while maintaining backward compatibility.

---

## Redis Persistence

```bash
export REDIS_URL=redis://localhost:6379
uvicorn main:app --port 8000
# On restart: price buffers restored, HMMs re-fitted automatically
```

Without `REDIS_URL`, all state is in-memory with zero overhead.

---

## GPU Acceleration

```bash
pip install pomegranate torch
GET /gpu/capabilities
# → {"active_device": "cuda", "hmm_backend": "GpuRegimeHMM (pomegranate)"}
```

Transparent fallback to hmmlearn on CPU — no code changes required.

---

## Notes

- Minimum **81 price bars** required (`min_bars = max(81, n_components × 20)`)
- `Model is not converging` warnings from hmmlearn are expected on short series — `_sanitize_model()` recovers automatically
- All position fractions are in `[0, 1]` portfolio-fraction space — multiply by notional externally
- VaR/CVaR are **historical (empirical)** — more robust for fat-tailed return distributions
- `covariance_type="diag"` used for rolling windows < 300 bars; switch to `"full"` with 500+ seed bars
- The 4-state HMM produces 4 vol + 4 trend posteriors; the obs builder aggregates these into 3 Markov buckets (indices 14–19), re-normalised after aggregation
