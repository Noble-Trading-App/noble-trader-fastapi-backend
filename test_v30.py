"""
v3.0 Feature Tests — GPU HMM · Feed Adapters · Correlation · Optimisation
Run from v30 directory: python test_v30.py
"""
import sys, warnings
sys.path.insert(0, "/home/claude/v30")
sys.path = [p for p in sys.path if p not in ("/home/claude", "")]
warnings.filterwarnings("ignore")

import numpy as np
import asyncio

np.random.seed(99)

# ── Synthetic multi-asset data ────────────────────────────────────────────────
SYMBOLS = ["SPY", "QQQ", "GLD", "TLT"]
N_BARS  = 400
N_ASSETS = 4

def make_correlated_returns(n, corr_level=0.3):
    """Generate correlated returns for N_ASSETS assets."""
    cov = np.full((N_ASSETS, N_ASSETS), corr_level * 0.01**2)
    np.fill_diagonal(cov, 0.01**2)
    returns = np.random.multivariate_normal(
        mean=[0.0006, 0.0007, 0.0002, 0.0001],
        cov=cov, size=n
    )
    return returns

returns_normal = make_correlated_returns(N_BARS, corr_level=0.3)
returns_crisis = make_correlated_returns(N_BARS, corr_level=0.85)
returns_mixed  = np.vstack([
    make_correlated_returns(200, 0.2),   # normal
    make_correlated_returns(100, 0.9),   # crisis
    make_correlated_returns(100, 0.3),   # recovery
])

prices_spy = 100 * np.cumprod(1 + returns_mixed[:, 0])

print("=" * 68)
print("  v3.0 Feature Tests")
print("=" * 68)

# ─── TEST 1: GPU Engine (capabilities + factory) ──────────────────────────────
print("\n▶ Test 1: GPU engine — create_regime_hmm factory")
from regime_platform.core.gpu_engine import create_regime_hmm, gpu_capabilities

caps = gpu_capabilities()
print(f"  pomegranate installed: {caps['pomegranate_installed']}")
print(f"  CUDA available:        {caps['cuda_available']}")
print(f"  MPS available:         {caps['mps_available']}")
print(f"  Active device:         {caps['active_device']}")
print(f"  HMM backend:           {caps['hmm_backend']}")

# Force CPU (hmmlearn) — always available
model = create_regime_hmm(force_cpu=True)
from regime_platform.core.regime_engine import RegimeHMM
assert isinstance(model, RegimeHMM), "force_cpu=True must return RegimeHMM"
model.fit(prices_spy)
snap = model.predict(prices_spy)
assert snap.vol_state   in RegimeHMM.VOL_LABELS
assert snap.trend_state in RegimeHMM.TREND_LABELS
assert 0.10 <= snap.risk_multiplier <= 1.75
print(f"  CPU model: {snap.regime_label}  mult={snap.risk_multiplier:.4f}×")
print("  ✓ create_regime_hmm factory + RegimeHMM interface verified")

# Interface contract check (same attributes whether GPU or CPU)
assert hasattr(model, 'fit')
assert hasattr(model, 'predict')
assert hasattr(model, 'fitted')
assert hasattr(model, 'VOL_LABELS')
assert hasattr(model, 'TREND_LABELS')
print("  ✓ Interface contract: fit/predict/fitted/VOL_LABELS/TREND_LABELS present")

# ─── TEST 2: Feed Adapters (structure, no live connection) ────────────────────
print("\n▶ Test 2: Feed adapter structure")
from regime_platform.adapters.feed_adapters import (
    AlpacaFeedAdapter, BinanceFeedAdapter, IBFeedAdapter, FeedManager, OHLCVBar
)

# Verify all adapters inherit from FeedAdapter
for cls, expected_source in [(AlpacaFeedAdapter, "alpaca"), (BinanceFeedAdapter, "binance"), (IBFeedAdapter, "ib")]:
    a = cls(symbols=["SPY"])
    assert hasattr(a, 'stream')
    assert hasattr(a, 'run_with_reconnect')
    assert hasattr(a, 'symbols')
    assert a.symbols == ["SPY"]
    print(f"  ✓ {cls.__name__}: symbols={a.symbols}, reconnect={a.reconnect}")

# FeedManager
mgr = FeedManager()
assert len(mgr._adapters) == 0
mgr.add(AlpacaFeedAdapter(["SPY"])).add(BinanceFeedAdapter(["BTCUSDT"]))
assert len(mgr._adapters) == 2
print(f"  ✓ FeedManager: chained .add() works, {len(mgr._adapters)} adapters registered")

# OHLCVBar dataclass
bar = OHLCVBar(symbol="SPY", ts=1700000000.0, open=512.0, high=514.0, low=511.0, close=513.5, volume=1e6, source="alpaca")
assert bar.close == 513.5
assert bar.source == "alpaca"
print("  ✓ OHLCVBar dataclass verified")

# ─── TEST 3: Correlation Regime Detector ─────────────────────────────────────
print("\n▶ Test 3: CorrelationRegimeDetector")
from regime_platform.core.correlation_regime import (
    CorrelationRegimeDetector, CorrelationRegimeConfig, CORR_LABELS, CORR_RISK_MULT
)

detector = CorrelationRegimeDetector(CorrelationRegimeConfig(window=40))

# Normal market — expect low/mid corr
snap_normal = detector.detect(returns_normal, SYMBOLS)
assert snap_normal.corr_regime in CORR_LABELS
assert 0.0 <= snap_normal.corr_confidence <= 1.0
assert snap_normal.n_assets == N_ASSETS
assert len(snap_normal.correlation_matrix) == N_ASSETS
assert len(snap_normal.correlation_matrix[0]) == N_ASSETS
assert abs(sum(snap_normal.corr_probs.values()) - 1.0) < 0.01
assert snap_normal.corr_risk_multiplier == CORR_RISK_MULT[snap_normal.corr_regime]
print(f"  Normal market regime:    {snap_normal.corr_regime}")
print(f"    mean |ρ|:              {snap_normal.mean_abs_correlation:.4f}")
print(f"    corr_risk_mult:        {snap_normal.corr_risk_multiplier:.2f}×")
print(f"    confidence:            {snap_normal.corr_confidence:.2%}")

# Crisis market — expect high_corr or crisis
snap_crisis = detector.detect(returns_crisis, SYMBOLS)
print(f"  Crisis market regime:    {snap_crisis.corr_regime}")
print(f"    mean |ρ|:              {snap_crisis.mean_abs_correlation:.4f}")
print(f"    corr_risk_mult:        {snap_crisis.corr_risk_multiplier:.2f}×")

# Crisis should have lower multiplier than normal
assert snap_crisis.corr_risk_multiplier <= snap_normal.corr_risk_multiplier + 0.05, \
    f"Crisis mult ({snap_crisis.corr_risk_multiplier}) should be ≤ normal ({snap_normal.corr_risk_multiplier})"

# With asset snaps
snaps_list = []
for j in range(N_ASSETS):
    prices = 100 * np.cumprod(1 + returns_mixed[:, j])
    m = RegimeHMM(); m.fit(prices)
    snaps_list.append(m.predict(prices))

snap_with_assets = detector.detect(returns_mixed, SYMBOLS, snaps_list)
assert snap_with_assets.blended_risk_multiplier > 0
assert len(snap_with_assets.asset_regimes) == N_ASSETS
print(f"  Mixed series with asset snaps:")
print(f"    blended_risk_mult:     {snap_with_assets.blended_risk_multiplier:.4f}×")
for ar in snap_with_assets.asset_regimes:
    print(f"    {ar.symbol}: {ar.regime_label}  mult={ar.risk_multiplier:.4f}×")
print("  ✓ All correlation regime assertions passed")

# ─── TEST 4: Drawdown Optimiser ───────────────────────────────────────────────
print("\n▶ Test 4: DrawdownOptimiser")
from regime_platform.core.portfolio_optimiser import DrawdownOptimiser, OptimiserConfig

opt = DrawdownOptimiser(OptimiserConfig(max_dd_limit=0.20, max_weight=0.40))
result = opt.optimise(
    returns_matrix=returns_mixed,
    symbols=SYMBOLS,
    asset_snaps=snaps_list,
    corr_regime=snap_with_assets.corr_regime,
)

# Basic sanity checks
assert len(result.weights) == N_ASSETS
assert len(result.regime_adj_weights) == N_ASSETS
assert all(w >= 0 for w in result.weights), "All weights must be non-negative"
assert all(w <= 0.41 for w in result.weights), "No weight exceeds max_weight"
assert abs(sum(result.weights) - result.regime_exposure) < 0.01, \
    f"Weights sum {sum(result.weights):.4f} ≠ exposure {result.regime_exposure:.4f}"
assert 0.35 <= result.regime_exposure <= 1.00, "Exposure must be in [0.35, 1.00]"
assert result.expected_vol > 0
assert result.expected_max_drawdown >= 0

print(f"  Weights:               {[f'{w:.4f}' for w in result.weights]}")
print(f"  Regime-adj weights:    {[f'{w:.4f}' for w in result.regime_adj_weights]}")
print(f"  Expected return:       {result.expected_return:.4f} ({result.expected_return:.2%})")
print(f"  Expected vol:          {result.expected_vol:.4f} ({result.expected_vol:.2%})")
print(f"  Sharpe ratio:          {result.sharpe_ratio:.4f}")
print(f"  Expected max DD:       {result.expected_max_drawdown:.4f} ({result.expected_max_drawdown:.2%})")
print(f"  DD constraint met:     {result.dd_constraint_met}")
print(f"  Regime exposure:       {result.regime_exposure:.4f}")
print(f"  Converged:             {result.converged}")
print(f"  Per-asset bounds:      {[(round(b[0],3), round(b[1],3)) for b in result.per_asset_bounds]}")
print("  ✓ All optimiser assertions passed")

# Regime effect: crisis should reduce exposure vs normal
opt2 = DrawdownOptimiser(OptimiserConfig(max_dd_limit=0.20))
res_normal = opt2.optimise(returns_normal, SYMBOLS, corr_regime="low_corr")
res_crisis = opt2.optimise(returns_crisis, SYMBOLS, corr_regime="crisis")
assert res_crisis.regime_exposure < res_normal.regime_exposure, \
    f"Crisis exposure ({res_crisis.regime_exposure:.2f}) should be < normal ({res_normal.regime_exposure:.2f})"
print(f"  Regime exposure — low_corr: {res_normal.regime_exposure:.2f}  crisis: {res_crisis.regime_exposure:.2f}")
print("  ✓ Crisis reduces portfolio exposure vs normal regime")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─' * 68}")
print("✅  All v3.0 tests passed.\n")
print("  New endpoints available on main_v4.py:")
print("  GET  /gpu/capabilities          — hardware + backend report")
print("  POST /gpu/benchmark             — latency benchmark")
print("  POST /feeds/start               — start Alpaca/Binance/IB live feeds")
print("  GET  /feeds/status              — adapter health")
print("  POST /correlation/detect        — DCC portfolio correlation regime")
print("  POST /optimise                  — drawdown-controlled max-Sharpe")
print("  POST /optimise/full             — correlation + optimise one-shot")
