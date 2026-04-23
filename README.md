# Dynamic Regime Risk Management Platform

A production-grade FastAPI web service for quantitative trading risk management. The platform detects market regimes with Hidden Markov Models and uses them to dynamically gate position sizes (Kelly Criterion) and risk limits (VaR/CVaR).

---

## Architecture

```
/
├── core/
│   ├── regime_engine.py     # Dual GaussianHMM (vol + trend)
│   ├── position_sizer.py    # Fractional Kelly + vol-scaling + regime gating
│   └── risk_manager.py      # VaR, CVaR, drawdown, stop/TP
├── models/
│   └── schemas.py           # Pydantic v2 request/response models
└── routers/
    ├── regime.py            # POST /regime/detect
    ├── sizing.py            # POST /size/kelly
    ├── risk.py              # POST /risk/analyse
    └── pipeline.py          # POST /analyse/full
main.py                      # FastAPI app factory + uvicorn entry point
requirements.txt
```

---

## Library Choices & Rationale

| Library | Role | Why |
|---------|------|-----|
| **FastAPI** | Web framework | Async, auto-docs, Pydantic v2 native, fastest Python HTTP |
| **hmmlearn** | Hidden Markov Models | Mature Gaussian HMM with Viterbi + posterior probabilities |
| **scipy** | Stats primitives | VaR/CVaR, normal distribution, optimisation scaffolding |
| **numpy** | Numerical core | Vectorised return/vol calculations, no pandas overhead in hot path |
| **pydantic v2** | Validation & serialisation | Fast Rust-core validation, discriminated unions, field constraints |
| **uvicorn** | ASGI server | Production-ready, supports reload + workers |

> **Why not pynamical/simupy?**  
> Both are simulation/dynamical-systems libraries rather than statistical inference or risk engines. `pynamical` is a chaos/bifurcation visualiser; `simupy` is a block-diagram ODE solver. For a trading risk service, `hmmlearn` + `scipy` give you real statistical rigour. If you want regime *simulation* for scenario generation, `simupy` can be layered in as a `/simulate` endpoint.

---

## Endpoints

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

### `POST /analyse/full` ⭐ Recommended
One-shot pipeline: sends prices → receives regime + sizing + risk in a single response.

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
uvicorn main:app --reload --port 8000

# Docs
open http://localhost:8000/docs
```

---

## Example Request

```bash
curl -X POST http://localhost:8000/analyse/full \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "prices": [100.0, 101.2, ...],   # 51+ prices
    "kelly_fraction": 0.5,
    "target_vol": 0.15,
    "base_risk_limit": 0.02
  }'
```

### Example Response (abbreviated)

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

## Regime → Risk Multiplier Map

| Vol \ Trend | Bear | Neutral | Bull |
|------------|------|---------|------|
| **Low**    | 0.75 | 1.28    | 1.50 |
| **Medium** | 0.50 | 0.85    | 1.00 |
| **High**   | 0.25 | 0.43    | 0.50 |

---

## Extending the Platform

- **Simulation endpoint** — add `simupy` for ODE-based regime transition simulation
- **Chaos analysis** — add `pynamical` for bifurcation/Lyapunov analysis of portfolio dynamics
- **Live data** — add a `/stream` WebSocket endpoint consuming real-time OHLCV feeds
- **Multi-asset** — parallelise HMM fitting with `concurrent.futures` or Celery workers
- **GPU acceleration** — replace `hmmlearn` with `pomegranate` (PyTorch backend) for GPU HMM

---

## Notes

- Minimum 51 price bars required (50 for HMM fit + 1 return)
- HMM convergence warnings are expected on short series; increase `n_iter` or bars for stability
- All position fractions are in `[0, 1]` portfolio-fraction space — scale by your capital externally
- VaR/CVaR are historical (empirical), not parametric — more accurate for fat-tailed asset returns
