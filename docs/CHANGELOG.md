## [2.2.0] — 2024-Q3

### Added — 24-Feature Observation Vector & Stateful Builder

#### `core/obs_builder.py` — NEW FILE
- `InferenceObservationBuilder` — stateful class that maintains a fitted `RegimeHMM`
  across calls, ensuring Markov features (indices 14–19) are always computed from
  live HMM posteriors, never from a uniform prior
- `F` — feature index namespace (single source of truth for all 24 indices)
- `Observation` — output dataclass with `.markov_features` slice property and
  `.is_markov_uniform(tol)` guard method
- `build(prices, high, low)` — full-window build; re-fits HMM every `refit_every` bars
- `build_from_tick(price, high, low)` — streaming single-bar variant
- `seed(prices, high, low)` — historical warm-up and initial HMM fit
- `update_recommended_f(f)` — injects current Kelly fraction for Masaniello feature

#### 24-Feature Vector

| Index | Feature | Notes |
|-------|---------|-------|
| 0 | `log_return_1bar` | 1-bar log return |
| 1 | `log_return_3bar` | 3-bar log return |
| 2 | `log_return_10bar` | 10-bar log return |
| 3 | `normalised_atr` | ATR(14) / price |
| 4 | `rolling_vol_20` | 20-bar σ (annualised) |
| 5 | `ema_distance` | (price − EMA20) / EMA20 |
| 6 | `hhll_score` | HH/HL structure score ∈ [−1, 1] |
| 7 | `vol_regime_percentile` | Vol percentile vs 252-bar history |
| 8 | `atr_vs_baseline` | ATR(14) / ATR(252-bar mean) |
| 9 | `vol_trend_slope` | Standardised 10-bar vol slope |
| 10–13 | `regime_prob_{0..3}` | Raw HMM state posteriors (unsorted) |
| **14** | **`vol_prob_low`** | P(low vol) — Markov feature |
| **15** | **`vol_prob_medium`** | P(med_low + med_high) — Markov feature |
| **16** | **`vol_prob_high`** | P(high vol) — Markov feature |
| **17** | **`trend_prob_down`** | P(strong_bear + ½·bear) — Markov feature |
| **18** | **`trend_prob_neutral`** | P(½·bear + ½·bull) — Markov feature |
| **19** | **`trend_prob_up`** | P(½·bull + strong_bull) — Markov feature |
| 20 | `regime_quality` | HMM confidence scalar |
| 21 | `state_confidence` | max(vol_probs) × max(trend_probs) |
| 22 | `masaniello_pressure` | f* × (1 − f*) — peaks at 0.25 when f=0.5 |
| 23 | `drawdown_factor` | Current DD from peak ∈ [−1, 0] |

#### `core/regime_engine.py` — Updated
- Added `_sanitize_model()` static method: fixes NaN `startprob_`, zero `transmat_`
  rows, NaN `means_`, and degenerate `_covars_` after EM convergence failure
- `fit()` calls `_sanitize_model()` on both sub-models after every EM run
- `predict()` adds NaN fallback: degenerate posteriors replaced with uniform, then
  re-normalised — guarantees finite output even on pathological price series

#### `test_obs_builder.py` — NEW FILE
- 7-test suite: shape, finite values, Markov non-uniformity, Masaniello response,
  drawdown sign, streaming tick path, and all-zero sentinel guard

### Fixed
- `RegimeHMM._sanitize_model()` writes to `_covars_` private attribute (not the
  property setter) to avoid hmmlearn's shape-validation error on `diag` covariance
- Uniform-prior bug documented and guarded: `is_markov_uniform()` method on
  `Observation` detects std < 0.05 across Markov features and warns

---

## [2.1.0] — 2024-Q3

### Changed — 4-State HMM Regime Detection

#### `core/regime_engine.py`
- **n_components: 3 → 4** for both the volatility and trend HMMs
- Volatility states: `low | med_low | med_high | high`  (was `low | medium | high`)
- Trend states: `strong_bear | bear | bull | strong_bull`  (was `bear | neutral | bull`)
- Volatile feature set expanded: vol HMM now uses 3 features (20-bar RV, 5-bar RV, |return|); trend HMM uses 3 features (return, 10-bar cum, 30-bar cum)
- Trend state labelling now sorts on feature index 2 (30-bar cumulative return) for cleaner strong_bull/strong_bear separation
- Min bars requirement: 80 (was 50) to support stable 4-state EM convergence
- Risk multiplier range extended: **[0.10, 1.75]** (was [0.25, 1.50])
- 4×4 = 16 distinct regime cells (was 9)

#### Downstream changes (label references only)
- `routers/regime.py`, `routers/pipeline.py` — label zips use `RegimeHMM.VOL_LABELS` / `RegimeHMM.TREND_LABELS`
- `routers/stream_rest.py`, `routers/stream_ws.py` — same label zip update
- `models/schemas.py`, `models/stream_schemas.py` — `min_length` / `ge` validators updated to 81
- `services/stream_session.py` — `MIN_PRICES_FOR_FIT` updated from 51 → 81

#### Added
- `test_platform_v4.py` — 4-state smoke test verifying 4 prob vectors, 16-cell multiplier table, and correct assertions

---

# Changelog

All notable changes to the Dynamic Regime Risk Management Platform are documented here.
Follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2024-Q2

### Added — Live Streaming Layer

#### `services/stream_session.py` — `StreamSession`
- Per-symbol stateful streaming engine with rolling price deque (configurable window, default 500 bars)
- Background HMM re-fit via `asyncio.run_in_executor` every N bars (default 50) — non-blocking
- 3-bar alert debounce: regime must persist for 3 consecutive bars before a `RegimeAlert` fires
- NaN/zero-price guard: invalid prices are filtered from the buffer before every HMM call
- Multiple concurrent subscriber queues per session (bounded at 500 items; slow consumers drop ticks)
- `RegimeAlert` dataclass with `severity` field: `info` | `warning` | `critical`

#### `services/registry.py` — `SessionRegistry`
- Singleton session store (`asyncio.Lock`-protected)
- `get_or_create(symbol, ...)` — idempotent session creation with per-session config
- `stats()` — returns live metrics for all sessions (ready, n_bars, refit_count, tick_count, last_regime)

#### `routers/stream_rest.py` — REST streaming endpoints
- `POST /stream/seed` — seed a symbol session with historical prices
- `POST /stream/tick` — push one price tick, receive `TickResponse` or buffering stub
- `POST /stream/ticks` — batch tick ingestion / backtest replay
- `GET /stream/sessions` — list all active sessions + stats
- `GET /stream/session/{symbol}` — single session status
- `DELETE /stream/session/{symbol}` — remove a session

#### `routers/stream_ws.py` — WebSocket + SSE endpoints
- `WS /ws/{symbol}` — full-duplex WebSocket: seed / tick / subscribe / ping message types
- `GET /sse/{symbol}` — SSE stream pushing `TickResponse` JSON on every bar
- `GET /sse/alerts` — global SSE broadcast for all regime-change alerts across all symbols
- 30-second heartbeat in subscriber mode; 15-second SSE comment heartbeat

#### `models/stream_schemas.py` — Streaming Pydantic models
- `SeedRequest`, `TickIngest`, `TickResponse`, `SessionStatus`, `WsMessage`

#### `main.py`
- Updated app factory including all streaming routers alongside existing batch routers

#### `test_streaming.py`
- Async smoke test: seed → 160-tick stream → subscriber queue → regime transition assertions

### Fixed

- `RegimeHMM` covariance type changed from `"full"` to `"diag"` for numerical stability on short rolling windows (< 300 bars)
- NaN/zero-price guard added to `_refit()` and `_compute_tick()` in `StreamSession`
- Test data continuity bug fixed (live price segment now correctly continues from seed tail)

---

## [1.0.0] — 2024-Q1

### Added — Batch Risk Management API

#### `core/regime_engine.py` — `RegimeHMM`
- Dual 3-state Gaussian HMM for volatility and trend regime classification
- Forward-backward posterior probabilities (not Viterbi hard assignments)
- Semantic state labelling by sorting component means — no manual mapping
- `RegimeSnapshot` dataclass with `risk_multiplier` property [0.25 – 1.50]

#### `core/position_sizer.py` — `PositionSizer`
- Full Kelly Criterion: `f* = μ / σ²` (continuous log-return formula)
- Fractional Kelly: `f* × kelly_fraction` (default 0.5)
- Volatility-scaled Kelly: `fractional_f × (target_vol / realised_annual_vol)`
- Regime-gated sizing: `vol_scaled_f × risk_multiplier`
- Automatic diagnostic notes (over-Kelly, negative Sharpe, vol mismatch, regime warnings)

#### `core/risk_manager.py` — `RiskManager`
- Historical (empirical) VaR at 95% and 99%
- CVaR / Expected Shortfall at 95% and 99%
- Maximum drawdown (peak-to-trough)
- Annualised return, volatility, Sortino ratio, Calmar ratio
- Regime-adjusted max loss limit: `base_risk_limit × risk_multiplier`
- Suggested stop-loss (2× CVaR95) and take-profit (3:1 reward/risk)
- Risk budget utilisation: `recommended_f / max_safe_f`

#### `models/schemas.py` — Pydantic v2 models
- `PricePayload`, `RegimeRequest/Response`, `SizeRequest/Response`, `RiskRequest/Response`, `FullAnalysisRequest/Response`
- Field-level validation: `min_length=51`, `ge`, `le`, `gt`, all-positive price guard

#### Routers
- `POST /regime/detect`
- `POST /size/kelly`
- `POST /risk/analyse`
- `POST /analyse/full` — one-shot full pipeline

#### Infrastructure
- `main.py` — FastAPI app factory with CORS middleware
- `requirements.txt` — pinned dependency versions
- `test_platform.py` — batch pipeline smoke test with synthetic multi-regime prices

---

## Roadmap

### Planned for v2.1
- `POST /simulate/{symbol}` — ODE-based regime transition simulation (simupy integration)
- `GET /portfolio` — multi-symbol aggregated regime + risk summary
- Redis Streams persistence for price buffers (survive process restarts)
- JWT authentication middleware for WebSocket handshake

### Planned for v3.0
- GPU-accelerated HMM via `pomegranate` (PyTorch backend)
- Real-time OHLCV feed adapters (Alpaca, Interactive Brokers, Binance)
- Multi-asset correlation regime detection
- Drawdown-controlled portfolio optimisation layer
