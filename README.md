# Dynamic Regime Risk Management Platform

A production-grade FastAPI web service for quantitative trading risk management. The platform detects market regimes with 4-state Hidden Markov Models, dynamically gates position sizes via the Kelly Criterion, computes real-time risk metrics (VaR/CVaR), streams live regime snapshots and alerts over WebSocket and SSE, and produces 24-feature observation vectors for downstream RL/ML policy layers.

---

## Version

**v3.1.0** — GPU HMM · Live feeds · Correlation regimes · Portfolio optimisation · Dynamic Masaniello+Kelly sizer

---

## Architecture

```
regime_platform/
├── core/
│   ├── regime_engine.py      # Dual 4-state GaussianHMM (vol + trend) + _sanitize_model
│   ├── obs_builder.py        # 24-feature InferenceObservationBuilder + F index namespace
│   ├── position_sizer.py     # Fractional Kelly + vol-scaling + regime gating
│   ├── risk_manager.py       # VaR, CVaR, drawdown, stop/TP
│   ├── simulator.py          # Markov-chain Monte Carlo regime transition simulator
│   ├── gpu_engine.py         # Pomegranate GPU HMM adapter + create_regime_hmm() factory
│   ├── correlation_regime.py # DCC multi-asset correlation regime detector
│   └── portfolio_optimiser.py # Drawdown-controlled regime-aware max-Sharpe optimiser
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
├── adapters/
│   └── feed_adapters.py      # Alpaca / Binance / IB live OHLCV feed adapters
├── auth/
│   └── jwt_auth.py           # JWT + API-key authentication, role-based access
└── services/
    ├── stream_session.py     # Per-symbol stateful streaming engine
    ├── registry.py           # Singleton session store
    ├── portfolio_service.py  # Multi-symbol regime + risk aggregation
    └── redis_persistence.py  # Optional Redis Streams price buffer persistence
main.py                       # v1 batch-only app factory
main_v2.py                    # v2 app factory (batch + streaming)
main_v3.py                    # v2.1 app factory (+ simulation, portfolio, auth, Redis)
main_v4.py                    # v3.0 app factory (+ GPU HMM, feeds, correlation, optimise)
requirements.txt
docs/
├── index.html                # Full documentation site (22 pages)
├── ws-client.html            # Interactive WebSocket test client
├── openapi.yaml              # OpenAPI 3.1 specification
├── CHANGELOG.md              # v1.0 → v2.2 version history
└── deployment.md             # Production operations guide
test_platform.py              # Batch pipeline smoke test
test_platform_v4.py           # 4-state HMM smoke test
test_streaming.py             # Streaming layer smoke test
test_obs_builder.py           # 24-feature observation builder test suite (7 tests)
test_v21.py                   # v2.1 feature test suite (5 tests)
test_v30.py                   # v3.0 feature test suite (4 tests)
```

---

## Library Choices & Rationale

| Library | Role | Why |
|---------|------|-----|
| **FastAPI** | Web framework | Async, auto-docs, Pydantic v2 native, fastest Python HTTP |
| **hmmlearn** | Hidden Markov Models | Mature `GaussianHMM` with forward-backward posterior probabilities |
| **scipy** | Stats primitives | VaR/CVaR, normal distribution, optimisation scaffolding |
| **numpy** | Numerical core | Vectorised return/vol calculations, no pandas overhead in hot path |
| **pydantic v2** | Validation & serialisation | Rust-core validation, discriminated unions, field constraints |
| **uvicorn** | ASGI server | Production-ready ASGI, supports reload + multi-worker |

> **Why not pynamical/simupy?**
> Both are simulation/dynamical-systems libraries, not statistical inference engines. `pynamical` is a chaos/bifurcation visualiser; `simupy` is a block-diagram ODE solver. Either could be layered in as a `/simulate` extension endpoint for scenario generation or regime transition modelling.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run (v2 — batch + streaming)
uvicorn main_v4:app --reload --port 8000

# Interactive Swagger docs
open http://localhost:8000/docs

# Local documentation site
open docs/index.html
```

---

## Endpoints

### Batch (one-shot historical analysis)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyse/full` | ⭐ Full pipeline: regime + sizing + risk in one request |
| `POST` | `/simulate/{symbol}` | Monte Carlo regime transition simulation |
| `GET` | `/gpu/capabilities` | GPU + HMM backend status |
| `POST` | `/gpu/benchmark` | Fit/predict latency benchmark |
| `POST` | `/feeds/start` | Start live Alpaca/Binance/IB feed |
| `GET` | `/feeds/status` | Feed adapter health + bar counts |
| `POST` | `/correlation/detect` | DCC multi-asset correlation regime |
| `POST` | `/optimise` | Drawdown-controlled portfolio weights |
| `POST` | `/optimise/full` | Correlation + optimise one-shot |
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
| `GET` | `/portfolio` | Multi-symbol aggregated regime + risk view |
| `POST` | `/auth/token` | Issue a JWT access token |
| `GET` | `/auth/me` | Return current user info from token |
| `WS` | `/ws/{symbol}` | Full-duplex WebSocket stream |
| `GET` | `/sse/{symbol}` | Server-Sent Events push stream (per symbol) |
| `GET` | `/sse/alerts` | SSE global regime-change alert broadcast |

---

## Regime Detection — 4-State HMM

Two independent 4-state Gaussian HMMs classify every price bar into a **volatility regime** and a **trend regime**. Both use forward-backward posterior probabilities (not Viterbi hard assignments), so confidence scores are smooth and differentiable. States are auto-labelled by sorting HMM component means — no manual mapping required.

### States

| Dimension | States | Features used |
|-----------|--------|---------------|
| Volatility | `low` · `med_low` · `med_high` · `high` | 20-bar RV, 5-bar RV, \|return\| |
| Trend | `strong_bear` · `bear` · `bull` · `strong_bull` | raw return, 10-bar cum return, 30-bar cum return |

### Risk Multiplier Table (4×4 = 16 regimes)

| Vol \ Trend   | Strong Bear | Bear  | Bull  | Strong Bull |
|--------------|-------------|-------|-------|-------------|
| **Low**      | 0.70×       | 1.05× | 1.54× | 1.75×       |
| **Med-Low**  | 0.48×       | 0.72× | 1.06× | 1.20×       |
| **Med-High** | 0.28×       | 0.42× | 0.62× | 0.70×       |
| **High**     | 0.14×       | 0.21× | 0.31× | 0.35×       |

Range: **[0.10, 1.75]**. The final Kelly position fraction is multiplied by this value before being recommended.

### Degenerate EM Recovery

`RegimeHMM._sanitize_model()` is called automatically after every HMM fit and corrects three failure modes common on short or strongly-structured price series:

| Failure | Cause | Fix |
|---------|-------|-----|
| NaN `startprob_` | All observations assigned to one state | Replace with uniform 1/n |
| Zero `transmat_` rows | State never visited during EM | Replace row with uniform 1/n |
| NaN/zero `_covars_` | Degenerate Gaussian emission | Replace bad entries with 1e-3 |

`predict()` adds a final NaN fallback — if posteriors are still NaN after sanitization, they are replaced with uniform and re-normalised, guaranteeing a finite observation vector.

---

## 24-Feature Observation Vector

`InferenceObservationBuilder` (in `core/obs_builder.py`) constructs a 24-feature observation vector per bar for downstream RL/ML policy layers.

### Feature Table

| Index | `F` constant | Feature | Range | Description |
|-------|-------------|---------|-------|-------------|
| 0 | `F.LOG_RET_1` | log_return_1bar | (−∞,+∞) | 1-bar log return |
| 1 | `F.LOG_RET_3` | log_return_3bar | (−∞,+∞) | 3-bar log return |
| 2 | `F.LOG_RET_10` | log_return_10bar | (−∞,+∞) | 10-bar log return |
| 3 | `F.NORMALISED_ATR` | normalised_atr | [0,+∞) | ATR(14) / price |
| 4 | `F.ROLLING_VOL_20` | rolling_vol_20 | [0,+∞) | 20-bar σ (annualised) |
| 5 | `F.EMA_DISTANCE` | ema_distance | (−∞,+∞) | (price − EMA20) / EMA20 |
| 6 | `F.HHLL_SCORE` | hhll_score | [−1,1] | Higher-high/lower-low structure score |
| 7 | `F.VOL_REGIME_PCT` | vol_regime_percentile | [0,1] | Vol percentile rank vs 252-bar history |
| 8 | `F.ATR_VS_BASELINE` | atr_vs_baseline | [0,+∞) | ATR(14) / ATR(252-bar mean) |
| 9 | `F.VOL_TREND_SLOPE` | vol_trend_slope | (−∞,+∞) | Standardised 10-bar vol slope |
| 10–13 | `F.REGIME_PROB_0..3` | regime_prob_{0..3} | [0,1] | Raw unsorted HMM state posteriors |
| **14** | **`F.VOL_PROB_LOW`** | **vol_prob_low** | [0,1] | **P(low vol) — Markov feature** |
| **15** | **`F.VOL_PROB_MEDIUM`** | **vol_prob_medium** | [0,1] | **P(med_low + med_high) — Markov feature** |
| **16** | **`F.VOL_PROB_HIGH`** | **vol_prob_high** | [0,1] | **P(high vol) — Markov feature** |
| **17** | **`F.TREND_PROB_DOWN`** | **trend_prob_down** | [0,1] | **P(strong_bear + ½·bear) — Markov feature** |
| **18** | **`F.TREND_PROB_NEUTRAL`** | **trend_prob_neutral** | [0,1] | **P(½·bear + ½·bull) — Markov feature** |
| **19** | **`F.TREND_PROB_UP`** | **trend_prob_up** | [0,1] | **P(½·bull + strong_bull) — Markov feature** |
| 20 | `F.REGIME_QUALITY` | regime_quality | [0,1] | HMM confidence scalar |
| 21 | `F.STATE_CONFIDENCE` | state_confidence | [0,1] | max(vol_probs) × max(trend_probs) |
| 22 | `F.MASANIELLO` | masaniello_pressure | [0,0.25] | f* × (1 − f*) — peaks at 0.25 when f=0.5 |
| 23 | `F.DRAWDOWN_FACTOR` | drawdown_factor | [−1,0] | (price − peak) / peak from running high watermark |

**Indices 14–19 are the Markov features.** They must always carry dynamic values from the fitted HMM — never uniform priors. If they show constant ~0.33 values, the uniform-prior bug is active.

### Usage

```python
from regime_platform.core.obs_builder import InferenceObservationBuilder, F

# Create stateful builder — maintains HMM state across bars
builder = InferenceObservationBuilder(
    window=200,          # rolling price buffer size
    refit_every=50,      # background HMM refit frequency (bars)
    recommended_f=0.35,  # Kelly fraction for Masaniello feature
)

# Seed with historical bars — required before any build/tick call
builder.seed(prices=close, high=high, low=low)
assert builder.is_ready

# Build full observation from window
obs = builder.build(prices=close, high=high, low=low)
assert obs.vector.shape == (24,)
assert not obs.is_markov_uniform()   # runtime guard: detects uniform-prior bug

# Access features by name (never by magic number)
vol_low  = obs.vector[F.VOL_PROB_LOW]     # index 14
drawdown = obs.vector[F.DRAWDOWN_FACTOR]  # index 23
masani   = obs.vector[F.MASANIELLO]       # index 22

# Streaming: one tick at a time
builder.update_recommended_f(0.42)        # inject current Kelly fraction
tick_obs = builder.build_from_tick(price=512.34, high=513.10, low=511.80)
if tick_obs:
    policy_action = model.predict(tick_obs.vector)
```

### ⚠ Uniform-Prior Bug

> If Markov features (indices 14–19) are constant (~0.33) across bars, the HMM state is not being injected and the observation is using a uniform prior.

**Always use `InferenceObservationBuilder`** (stateful class), never a standalone `build_observation()` function. Call `obs.is_markov_uniform()` to detect this at runtime.

| Pattern | Markov features | Use |
|---------|----------------|-----|
| `InferenceObservationBuilder.build()` | ✅ Dynamic from fitted HMM | Always |
| Standalone `build_observation()` | ❌ Uniform [0.25, 0.25, 0.25, 0.25] | Never |

### Full Pipeline Integration

```python
from regime_platform.core.obs_builder import InferenceObservationBuilder, F
from regime_platform.core.position_sizer import PositionSizer, PositionRequest

builder = InferenceObservationBuilder(window=200)
sizer   = PositionSizer()

builder.seed(prices, high, low)

for bar in live_feed:
    obs = builder.build_from_tick(bar.close, bar.high, bar.low)
    if obs is None:
        continue  # still warming up

    result = sizer.size(PositionRequest(
        returns=returns,
        kelly_fraction=0.5,
        regime=builder.last_snap,
    ))

    builder.update_recommended_f(result.recommended_f)  # keep Masaniello current

    assert not obs.is_markov_uniform()
    policy_action = model.predict(obs.vector)
```

---

## Kelly Position Sizing

Position sizes use the continuous-returns Kelly formula `f* = μ / σ²`, passed through three scaling layers:

| Stage | Field | Formula |
|-------|-------|---------|
| 1. Fractional Kelly | `fractional_f` | `full_kelly_f × kelly_fraction` (default 0.5) |
| 2. Vol scaling | `vol_scaled_f` | `fractional_f × (target_vol / realised_annual_vol)` |
| 3. Regime gating | `regime_gated_f` | `vol_scaled_f × risk_multiplier` |

`regime_gated_f` equals `recommended_f` — multiply by portfolio notional to get dollar size.

---

## Streaming Workflow

### 1. Seed the session

```bash
curl -X POST http://localhost:8000/stream/seed \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "prices": [...200+ bars], "kelly_fraction": 0.5}'
```

### 2a. REST tick ingestion

```bash
curl -X POST http://localhost:8000/stream/tick \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "price": 512.34}'
```

### 2b. WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/SPY');

ws.send(JSON.stringify({ type: "seed", symbol: "SPY", prices: [...] }));
ws.send(JSON.stringify({ type: "tick", symbol: "SPY", price: 512.34 }));
ws.onmessage = e => {
  const tick = JSON.parse(e.data);
  console.log(tick.regime_label, tick.recommended_f, tick.var_95);
};
```

### 2c. SSE

```javascript
const feed   = new EventSource('http://localhost:8000/sse/SPY');
const alerts = new EventSource('http://localhost:8000/sse/alerts');
alerts.onmessage = e => {
  const a = JSON.parse(e.data);
  // severity: "info" | "warning" | "critical"
  if (a.severity === 'critical') triggerRiskAlert(a);
};
```

---

## Example Responses

### `POST /analyse/full`

```json
{
  "symbol": "SPY",
  "regime": {
    "vol_state": "low",
    "trend_state": "strong_bull",
    "regime_label": "low_vol_strong_bull",
    "confidence": 0.8234,
    "risk_multiplier": 1.75
  },
  "sizing": {
    "full_kelly_f": 0.8421,
    "fractional_f": 0.4211,
    "vol_scaled_f": 0.3247,
    "regime_gated_f": 0.5682,
    "recommended_f": 0.5682,
    "sharpe_ratio": 1.34,
    "fraction_type": "50% Kelly",
    "notes": []
  },
  "risk": {
    "var_95": 0.0187,
    "cvar_95": 0.0263,
    "max_drawdown": -0.1124,
    "sortino_ratio": 1.82,
    "calmar_ratio": 1.97,
    "suggested_stop": -0.0526,
    "suggested_tp": 0.0789,
    "risk_budget_used": 0.61
  }
}
```

### Streaming tick

```json
{
  "symbol": "SPY",
  "ts": 1710000512.0,
  "price": 512.34,
  "n_bars": 247,
  "regime_label": "low_vol_strong_bull",
  "vol_state": "low",
  "trend_state": "strong_bull",
  "vol_probs":   { "low": 0.78, "medium": 0.18, "high": 0.04 },
  "trend_probs": { "bear": 0.03, "neutral": 0.09, "bull": 0.88 },
  "confidence": 0.6864,
  "risk_multiplier": 1.75,
  "recommended_f": 0.5682,
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

## StreamSession Internals

Each symbol's `StreamSession` has three concurrent responsibilities:

**Tick path (~7 ms/tick)** — predict regime posteriors → Kelly sizing → VaR/CVaR → push to all subscriber queues.

**Background HMM refit** (every 50 bars) — runs via `asyncio.run_in_executor`, never blocking the event loop. Includes `_sanitize_model()` for degenerate parameter recovery.

**Alert debounce** — a 3-bar hysteresis window prevents noisy micro-transitions from flooding consumers. Alerts carry `severity`: `info` / `warning` / `critical`.

Multiple concurrent WebSocket clients and SSE consumers per symbol are supported. Each subscriber gets a bounded `asyncio.Queue` (500 items). Slow consumers drop ticks rather than blocking the tick path.

---

## Running Tests

```bash
python test_platform.py       # batch pipeline
python test_platform_v4.py    # 4-state HMM + 16-cell multiplier table
python test_streaming.py      # seed → 160 ticks → subscriber → regime transitions
python test_obs_builder.py    # 7-test observation builder suite
```

---

## v2.1 Features

### Regime Simulation — `POST /simulate/{symbol}`

Fits a 4-state HMM, then builds a 16×16 Kronecker combined transition matrix (`T_combined = T_vol ⊗ T_trend`) and runs Monte Carlo paths forward. Returns a price fan (p5/p25/median/p75/p95), expected risk multiplier per step, terminal VaR/CVaR, and regime occupancy probabilities.

```bash
curl -X POST http://localhost:8000/simulate/SPY \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"prices": [...], "horizon": 20, "n_paths": 500}'
```

### Portfolio Aggregation — `GET /portfolio`

Reads all active `StreamSession`s and returns per-symbol regime + risk alongside portfolio-level analytics. Risk flags: `high_risk_count`, `concentration_flag`, `regime_divergence_flag`. Portfolio VaR uses the independence assumption: √(ΣVaR²).

```bash
curl "http://localhost:8000/portfolio?symbols=SPY,QQQ,GLD" \
  -H "Authorization: Bearer <token>"
```

### JWT Authentication

Configure via environment variables. Set `AUTH_ENABLED=false` in development.

```bash
# Configure
export JWT_SECRET_KEY="your-signing-secret"
export AUTH_USERS="admin:pass:admin,trader:pass123:trader"
export API_KEYS="svc-key-1"

# Issue a token
curl -X POST http://localhost:8000/auth/token -d "username=trader&password=pass123"

# Use token
curl -H "Authorization: Bearer <jwt>" http://localhost:8000/portfolio

# WebSocket with token
ws://localhost:8000/ws/SPY?token=<jwt>
```

Roles: `admin` (full access) · `trader` (read + write) · `viewer` (read-only).

### Redis Persistence

Set `REDIS_URL` to enable. Zero overhead when not set — all methods are no-ops.

```bash
export REDIS_URL=redis://localhost:6379
uvicorn main_v3:app --port 8000
# On restart: buffers automatically restored, HMMs re-fitted
```

---

## Dynamic Masaniello Position Sizer

`core/position_sizer.py` now implements two complementary sizing paths in the same module:

### Path A — Portfolio-fraction Kelly (original, unchanged)
Stateless. Input: historical returns + optional `RegimeSnapshot`. Output: `KellyResult` with `recommended_f` ∈ [0, 1].

### Path B — Dynamic Masaniello (new, stateful)
Stateful batch tracker. Input: equity, stop distance, point value, win probability, R/R, regime snapshot. Output: `PositionSizeResult` with contracts.

**Formula:** `f_i = β × (0.5 + M_i) × Q_i × DD_i × V_i`

| Factor | Formula | Clamp | Purpose |
|--------|---------|-------|---------|
| β | `base_risk` (default 0.50%) | — | Risk anchor |
| M_i | `(W − w) / (N − i + 1)` | [0.0, 1.5] | Batch urgency — wins needed ÷ trades left |
| Q_i | `prob_factor × regime_factor × conf_factor` | product | Edge × regime × confidence |
| DD_i | `clip(1 − dd/max_dd, 0.25, 1.0)` | [0.25, 1.0] | Drawdown protection, floor at 0.25 |
| V_i | `clip(ATR_base/ATR_current, 0.5, 1.5)` | [0.5, 1.5] | Volatility adjustment |

```python
from regime_platform.core.position_sizer import DynamicMasanielloSizer, SizingConfig

sizer = DynamicMasanielloSizer(SizingConfig(
    base_risk=0.005, batch_size=5, target_wins=3
))
sizer.batch.peak_equity = 100_000

result = sizer.size_from_snapshot(
    snap=snap,                   # RegimeSnapshot from HMM
    equity=100_000,
    stop_distance_price=2.50,
    point_value=50.0,
    p_win=0.62, reward_risk=2.5,
    atr_baseline=12.0, atr_current=10.0,
)
print(result.summary())
# [✓ ALLOWED]  f=0.550%  $550  units=5.50  contracts=5  ...

sizer.batch.record(result, outcome=True, pnl=625.0)
sizer.batch.reset(equity=100_625)  # next batch
```

See `test_position_sizer.py` for the full 95-test suite including Monte Carlo validation.

---

## v3.0 Features

### GPU-Accelerated HMM — `create_regime_hmm()`

```python
from regime_platform.core.gpu_engine import create_regime_hmm, gpu_capabilities

print(gpu_capabilities())
# → {"active_device": "cuda", "hmm_backend": "GpuRegimeHMM (pomegranate)"}

model = create_regime_hmm()   # GPU if available, CPU fallback
model.fit(prices)             # same interface as RegimeHMM
```

Install: `pip install pomegranate torch`. No code changes when GPU unavailable.

### Live Feed Adapters — `POST /feeds/start`

```bash
# 1. Seed symbols first
POST /stream/seed  {"symbol": "SPY", "prices": [...]}

# 2. Start Alpaca feed (bars auto-routed into sessions)
POST /feeds/start
[{"source": "alpaca", "symbols": ["SPY","QQQ"], "bar_size": "1Min"},
 {"source": "binance", "symbols": ["BTCUSDT"], "bar_size": "1m"}]

# 3. Monitor
GET /feeds/status
```

| Source | Asset classes | Install |
|--------|--------------|---------|
| `alpaca` | US equities, crypto | `pip install alpaca-py` |
| `binance` | All crypto spot pairs | `pip install websockets` |
| `ib` | Equities, futures, FX | `pip install ib_async` |

### Multi-Asset Correlation Regime — `POST /correlation/detect`

Detects portfolio-level correlation regimes via DCC (rolling Pearson + Gaussian HMM on upper-triangle features). Catches the crash regime where diversification collapses — something per-asset HMMs miss.

| Regime | Mean |ρ| | Multiplier | When |
|--------|--------|-----------|------|
| `low_corr` | < 0.20 | 1.00× | Normal |
| `mid_corr` | 0.20–0.50 | 0.85× | Moderate stress |
| `high_corr` | 0.50–0.75 | 0.60× | Pre-crisis |
| `crisis` | > 0.75 | 0.35× | Crash |

### Drawdown-Controlled Optimisation — `POST /optimise/full`

```bash
POST /optimise/full
{
  "symbols": ["SPY", "QQQ", "GLD", "TLT"],
  "returns_matrix": [[r_spy, r_qqq, r_gld, r_tlt], ...],
  "max_dd_limit": 0.15,
  "max_weight": 0.40
}
# Returns correlation regime + regime-adjusted optimal weights
# with drawdown constraint enforced via Cornish-Fisher CVaR proxy
```

---

## Extending the Platform

- **Simulation** — add `simupy` for ODE-based regime transition simulation as a `/simulate` endpoint
- **Chaos analysis** — add `pynamical` for bifurcation/Lyapunov analysis of portfolio dynamics
- **Multi-asset portfolio** — the `SessionRegistry` supports concurrent per-symbol sessions; add a `/portfolio` aggregation layer
- **GPU HMM** — replace `hmmlearn` with `pomegranate` (PyTorch backend) for GPU-accelerated EM
- **Persistence** — swap the `deque` price buffer for Redis Streams to survive process restarts
- **Authentication** — add FastAPI JWT middleware to the WebSocket handshake
- **RL integration** — pipe `InferenceObservationBuilder.build_from_tick()` output directly into a policy network

---

## Notes

- Minimum **81 price bars** required to fit the 4-state HMM (`min_bars = max(81, n_components × 20)`)
- `Model is not converging` warnings from hmmlearn are expected on short series — `_sanitize_model()` recovers automatically
- All position fractions are in `[0, 1]` portfolio-fraction space — multiply by notional capital externally
- VaR/CVaR are **historical (empirical)**, not parametric — more robust for fat-tailed return distributions
- `covariance_type="diag"` is used for numerical stability on rolling windows under 300 bars; switch to `"full"` with 500+ seed bars
- The 4-state HMM produces 4 vol and 4 trend posteriors; the observation builder aggregates these into 3 Markov buckets (indices 14–19) matching the original spec, re-normalising after aggregation
