"""
Smoke test — runs the full pipeline on synthetic price data.
Run from /home/claude:  python test_platform.py
"""

import sys, os
sys.path.insert(0, "/home/claude")

import numpy as np
from regime_platform.core.regime_engine import RegimeHMM
from regime_platform.core.position_sizer import PositionSizer, PositionRequest
from regime_platform.core.risk_manager import RiskManager

# ── Generate synthetic multi-regime prices ────────────────────────────────────
np.random.seed(42)
n = 500
returns = np.concatenate([
    np.random.normal(0.0008, 0.008, 150),   # low-vol bull
    np.random.normal(-0.001, 0.025, 100),   # high-vol bear
    np.random.normal(0.0002, 0.012, 150),   # medium-vol neutral
    np.random.normal(0.0012, 0.007, 100),   # low-vol bull again
])
prices = 100 * np.cumprod(1 + returns)

print("=" * 60)
print("  Dynamic Regime Risk Management Platform — Smoke Test")
print("=" * 60)

# 1. Regime
model = RegimeHMM()
model.fit(prices)
snap  = model.predict(prices)
print(f"\n📊 REGIME DETECTION")
print(f"   Regime:        {snap.regime_label}")
print(f"   Confidence:    {snap.confidence:.2%}")
print(f"   Risk Mult:     {snap.risk_multiplier:.2f}×")
print(f"   Vol  probs:    low={snap.vol_probs[0]:.2f}  med={snap.vol_probs[1]:.2f}  high={snap.vol_probs[2]:.2f}")
print(f"   Trend probs:   bear={snap.trend_probs[0]:.2f}  neutral={snap.trend_probs[1]:.2f}  bull={snap.trend_probs[2]:.2f}")

# 2. Position Sizing
sizer  = PositionSizer()
result = sizer.size(PositionRequest(
    returns=returns.tolist(),
    kelly_fraction=0.5,
    target_vol=0.15,
    regime=snap,
))
print(f"\n💰 POSITION SIZING  (50% Kelly)")
print(f"   Full Kelly f:   {result.full_kelly_f:.4f}  ({result.full_kelly_f:.1%})")
print(f"   Fractional f:   {result.fractional_f:.4f}")
print(f"   Vol-Scaled f:   {result.vol_scaled_f:.4f}")
print(f"   Regime-Gated f: {result.regime_gated_f:.4f}  ← recommended")
print(f"   Sharpe ratio:   {result.sharpe_ratio:.2f}")
for note in result.notes:
    print(f"   ⚠  {note}")

# 3. Risk
riskman = RiskManager()
risk    = riskman.analyse(returns=returns, recommended_f=result.recommended_f, regime=snap)
print(f"\n🛡  RISK METRICS")
print(f"   1-day VaR 95%:  {risk.var_95:.4f}  ({risk.var_95:.2%})")
print(f"   1-day VaR 99%:  {risk.var_99:.4f}  ({risk.var_99:.2%})")
print(f"   CVaR 95%:       {risk.cvar_95:.4f}")
print(f"   Max Drawdown:   {risk.max_drawdown:.2%}")
print(f"   Annual Vol:     {risk.annual_vol:.2%}")
print(f"   Annual Return:  {risk.annual_return:.2%}")
print(f"   Sortino:        {risk.sortino_ratio:.2f}")
print(f"   Calmar:         {risk.calmar_ratio:.2f}")
print(f"   Suggested Stop: {risk.suggested_stop:.4f}")
print(f"   Suggested TP:   {risk.suggested_tp:.4f}")
print(f"   Risk Budget:    {risk.risk_budget_used:.2f}×")

print(f"\n✅  All tests passed.\n")
