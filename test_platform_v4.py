"""
Smoke test — 4-state HMM pipeline.
Run from /home/claude:  python test_platform_v4.py
"""

import sys, os
sys.path.insert(0, "/home/claude")

import numpy as np
from rp4.core.regime_engine import RegimeHMM
from rp4.core.position_sizer import PositionSizer, PositionRequest
from rp4.core.risk_manager import RiskManager

# ── Generate synthetic 4-regime prices ───────────────────────────────────────
np.random.seed(42)

def seg(n, mu, sigma, start):
    r = np.random.normal(mu, sigma, n)
    return list(start * np.cumprod(1 + r))

s1 = seg(150, 0.0012, 0.005, 100.0)           # low_vol strong_bull
s2 = seg(120, 0.0004, 0.011, s1[-1])           # med_low_vol bull
s3 = seg(100, -0.0010, 0.020, s2[-1])          # med_high_vol bear
s4 = seg(130, -0.0020, 0.032, s3[-1])          # high_vol strong_bear
prices = np.array(s1 + s2 + s3 + s4)

print("=" * 64)
print("  Dynamic Regime Risk Platform — 4-State HMM Smoke Test")
print("=" * 64)
print(f"\n  Price bars: {len(prices)}  |  Range: {prices.min():.2f} – {prices.max():.2f}")

# 1. Regime
model = RegimeHMM()          # n_components=4 by default
model.fit(prices)
snap  = model.predict(prices)

print(f"\n📊 REGIME DETECTION  (4×4 states)")
print(f"   Regime:         {snap.regime_label}")
print(f"   Confidence:     {snap.confidence:.2%}")
print(f"   Risk Mult:      {snap.risk_multiplier:.4f}×")
print(f"\n   Vol  probs:")
for lbl, p in zip(RegimeHMM.VOL_LABELS, snap.vol_probs):
    bar = "█" * int(p * 30)
    print(f"     {lbl:<12} {p:.3f}  {bar}")
print(f"\n   Trend probs:")
for lbl, p in zip(RegimeHMM.TREND_LABELS, snap.trend_probs):
    bar = "█" * int(p * 30)
    print(f"     {lbl:<14} {p:.3f}  {bar}")

assert len(snap.vol_probs)   == 4, f"Expected 4 vol probs, got {len(snap.vol_probs)}"
assert len(snap.trend_probs) == 4, f"Expected 4 trend probs, got {len(snap.trend_probs)}"
assert snap.vol_state   in RegimeHMM.VOL_LABELS,   f"Bad vol state: {snap.vol_state}"
assert snap.trend_state in RegimeHMM.TREND_LABELS, f"Bad trend state: {snap.trend_state}"
assert abs(sum(snap.vol_probs)   - 1.0) < 1e-4, "Vol probs don't sum to 1"
assert abs(sum(snap.trend_probs) - 1.0) < 1e-4, "Trend probs don't sum to 1"
print("\n   ✓ 4 vol states, 4 trend states, probs sum to 1.0")

# 2. All 16 regime combinations reachable via risk_multiplier
print(f"\n📐 RISK MULTIPLIER TABLE  (4×4 = 16 cells)")
all_mults = []
for vs in RegimeHMM.VOL_LABELS:
    row = []
    for ts in RegimeHMM.TREND_LABELS:
        snap_tmp = RegimeHMM.__new__(RegimeHMM)
        from rp4.core.regime_engine import RegimeSnapshot
        s = RegimeSnapshot(
            vol_probs=[0.0]*4, trend_probs=[0.0]*4,
            vol_state=vs, trend_state=ts,
            regime_label=f"{vs}_vol_{ts}",
            confidence=1.0, n_bars_fitted=500
        )
        row.append(s.risk_multiplier)
        all_mults.append(s.risk_multiplier)
    row_str = "  ".join(f"{v:.2f}" for v in row)
    print(f"   {vs:<12} {row_str}")
print(f"\n   Range: {min(all_mults):.2f}× – {max(all_mults):.2f}×")
assert min(all_mults) >= 0.10
assert max(all_mults) <= 1.75
print("   ✓ All 16 cells within [0.10, 1.75]")

# 3. Position Sizing
sizer  = PositionSizer()
returns = np.diff(np.log(prices))
result = sizer.size(PositionRequest(
    returns=returns.tolist(),
    kelly_fraction=0.5,
    target_vol=0.15,
    regime=snap,
))
print(f"\n💰 POSITION SIZING  (50% Kelly)")
print(f"   Full Kelly f:    {result.full_kelly_f:.4f}  ({result.full_kelly_f:.1%})")
print(f"   Fractional f:    {result.fractional_f:.4f}")
print(f"   Vol-Scaled f:    {result.vol_scaled_f:.4f}")
print(f"   Regime-Gated f:  {result.regime_gated_f:.4f}  ← recommended")
print(f"   Sharpe ratio:    {result.sharpe_ratio:.2f}")
for note in result.notes:
    print(f"   ⚠  {note}")

# 4. Risk
riskman = RiskManager()
risk    = riskman.analyse(returns=returns, recommended_f=result.recommended_f, regime=snap)
print(f"\n🛡  RISK METRICS")
print(f"   1-day VaR 95%:   {risk.var_95:.4f}  ({risk.var_95:.2%})")
print(f"   CVaR 95%:        {risk.cvar_95:.4f}")
print(f"   Max Drawdown:    {risk.max_drawdown:.2%}")
print(f"   Annual Vol:      {risk.annual_vol:.2%}")
print(f"   Sortino:         {risk.sortino_ratio:.2f}")
print(f"   Calmar:          {risk.calmar_ratio:.2f}")
print(f"   Suggested Stop:  {risk.suggested_stop:.4f}")
print(f"   Suggested TP:    {risk.suggested_tp:.4f}")

print(f"\n✅  All 4-state HMM tests passed.\n")
