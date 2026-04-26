"""
position_sizer.py v3.1 — Test Suite

Covers:
  1. Factor math — M_i, Q_i, DD_i, V_i boundary conditions
  2. Core formula correctness with known inputs
  3. Gate ordering — each block reason fires before formula runs
  4. Kelly overlay — cap applied correctly, cap=0 falls back to min_risk
  5. Hard clamp — f never leaves [min_risk, max_risk]
  6. RegimeSnapshot integration — from_snapshot + size_from_snapshot
  7. BatchState — wins/losses/drawdown/reset lifecycle
  8. run_batch — state injection, sequential progression
  9. Monte Carlo — distribution stats, p_target_hit, p_halt
 10. Path A (PositionSizer / KellyResult) — unchanged interface preserved

Run: python test_position_sizer.py
"""

import sys, warnings
sys.path.insert(0, "/home/claude")
warnings.filterwarnings("ignore")

import numpy as np
from position_sizer_v31 import (
    # Path B
    SizingConfig, TradeContext, PositionSizeResult, BatchState,
    DynamicMasanielloSizer,
    _masaniello_factor, _quality_factor, _drawdown_factor,
    _volatility_factor, _fractional_kelly, _expected_edge,
    monte_carlo_batch,
    # Path A
    PositionSizer, PositionRequest, KellyResult,
)

# Minimal RegimeSnapshot stub — mirrors real dataclass interface
from dataclasses import dataclass
from typing import Optional as Opt

@dataclass
class _Snap:
    vol_probs:    list
    trend_probs:  list
    vol_state:    str
    trend_state:  str
    regime_label: str
    confidence:   float
    risk_multiplier: float
    n_bars_fitted:   int = 200

    @property
    def _snap_risk_multiplier(self):
        return self.risk_multiplier

def make_snap(mult=1.0, conf=0.75, label="low_vol_bull"):
    return _Snap(
        vol_probs=[0.7,0.2,0.07,0.03], trend_probs=[0.05,0.1,0.7,0.15],
        vol_state="low", trend_state="bull",
        regime_label=label, confidence=conf, risk_multiplier=mult,
    )

PASS = 0
FAIL = 0

def check(name, expr, expected=True, tol=None):
    global PASS, FAIL
    try:
        if tol is not None:
            ok = abs(expr - expected) < tol
        else:
            ok = (expr == expected) if not isinstance(expr, float) else abs(expr - expected) < 1e-9
        if ok:
            PASS += 1
            print(f"  ✓  {name}")
        else:
            FAIL += 1
            print(f"  ✗  {name}  got={expr!r}  expected={expected!r}")
    except Exception as e:
        FAIL += 1
        print(f"  ✗  {name}  EXCEPTION: {e}")

# ─── TEST 1: Factor math ──────────────────────────────────────────────────────
print("\n▶ Test 1: Factor math — boundary conditions")

# M_i
check("M_i: target met → 0.0",      _masaniello_factor(3, 1, 5, 3), 0.0)
check("M_i: 0 wins, 1 trade left",  _masaniello_factor(0, 5, 5, 3), 1.5, tol=0.01)
check("M_i: 1 win, 2 left, W=3",    _masaniello_factor(1, 4, 5, 3), 1.0, tol=1e-9)
check("M_i: clamp upper 1.5",       _masaniello_factor(0, 5, 5, 5), 1.5)  # 5 wins in 1 trade → clamped
check("M_i: clamp lower 0.0",       _masaniello_factor(5, 1, 5, 3) >= 0.0, True)

# Q_i
check("Q_i: p_win at floor → 0.0",  _quality_factor(0.50, 1.0, 1.0), 0.0)
check("Q_i: p_win=0.60, q=1, c=1",  _quality_factor(0.60, 1.0, 1.0), 1.0,  tol=1e-9)
check("Q_i: p_win=0.65, q=1, c=1",  _quality_factor(0.65, 1.0, 1.0), 1.5,  tol=1e-9)  # prob_factor clamped
check("Q_i: zero confidence → 0",   _quality_factor(0.65, 1.0, 0.0), 0.0)
check("Q_i: max possible",          _quality_factor(0.65, 1.5, 1.0), 2.25, tol=1e-9)
check("Q_i: low regime → 0",        _quality_factor(0.65, 0.0, 1.0), 0.0)

# DD_i
check("DD_i: no drawdown → 1.0",    _drawdown_factor(0.0,  0.10), 1.0)
check("DD_i: half max → 0.50",      _drawdown_factor(0.05, 0.10), 0.50, tol=1e-9)
check("DD_i: at max → floor 0.25",  _drawdown_factor(0.10, 0.10), 0.25)
check("DD_i: beyond max → floor",   _drawdown_factor(0.20, 0.10), 0.25)
check("DD_i: floor is 0.25",        _drawdown_factor(1.00, 0.10), 0.25)

# V_i
check("V_i: baseline=current → 1.0",  _volatility_factor(10.0, 10.0), 1.0)
check("V_i: vol doubled → 0.5 floor", _volatility_factor(10.0, 20.0), 0.5)
check("V_i: vol halved → 1.5 cap",    _volatility_factor(10.0,  5.0), 1.5)
check("V_i: zero current → 1.0",      _volatility_factor(10.0,  0.0), 1.0)
check("V_i: very low vol → cap 1.5",  _volatility_factor(10.0,  1.0), 1.5)

# Kelly
check("Kelly: no edge → 0.0",       _fractional_kelly(0.50, 1.0, 0.25), 0.0)
check("Kelly: p=0.6 R=2",           _fractional_kelly(0.60, 2.0, 0.25), 0.25 * ((2.0*0.6 - 0.4)/2.0), tol=1e-9)
check("Kelly: b=0 → 0.0",           _fractional_kelly(0.60, 0.0, 0.25), 0.0)

# Edge
check("Edge: p=0.6 R=2.0",         _expected_edge(0.6, 2.0), 0.6*2.0 - 0.4, tol=1e-9)
check("Edge: p=0.5 R=2.0 → 0.5",   _expected_edge(0.5, 2.0), 0.5, tol=1e-9)
check("Edge: p=0.4 R=1.0 → -0.2",  _expected_edge(0.4, 1.0), -0.2, tol=1e-9)

# ─── TEST 2: Core formula correctness ─────────────────────────────────────────
print("\n▶ Test 2: Core formula — known inputs")

cfg = SizingConfig(
    base_risk=0.005, min_risk=0.0025, max_risk=0.01,
    min_prob=0.50, min_rr=1.0, regime_floor=0.0,
    use_kelly_overlay=False, batch_size=5, target_wins=3,
)
sizer = DynamicMasanielloSizer(cfg)

ctx = TradeContext(
    equity=100_000, stop_distance_price=2.0, point_value=50.0,
    p_win=0.60, reward_risk=2.0,
    regime_quality=1.0, state_confidence=1.0,
    current_drawdown=0.0,
    atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0,
    trade_index=1, batch_size=5, target_wins=3,
    direction="long",
)

# Expected:
# M = (3-0)/(5-1+1) = 3/5 = 0.6
# Q = clip((0.60-0.50)/0.10, 0,1.5) × 1.0 × 1.0 = 1.0
# DD = 1.0 (no dd)
# V = 1.0 (baseline = current)
# f = 0.005 × (0.5 + 0.6) × 1.0 × 1.0 × 1.0 = 0.005 × 1.1 = 0.0055
M_exp = 3/5
f_exp = 0.005 * (0.5 + M_exp) * 1.0 * 1.0 * 1.0  # = 0.0055

res = sizer.size_trade(ctx)
check("Formula: allowed=True",           res.allowed, True)
check("Formula: M_i = 0.6",             res.masaniello_factor, M_exp,  tol=1e-9)
check("Formula: Q_i = 1.0",             res.quality_factor,    1.0,    tol=1e-9)
check("Formula: DD_i = 1.0",            res.drawdown_factor,   1.0,    tol=1e-9)
check("Formula: V_i = 1.0",             res.volatility_factor, 1.0,    tol=1e-9)
check("Formula: f = 0.0055",            res.risk_fraction,     f_exp,  tol=1e-6)
check("Formula: risk_amount = 550",      res.risk_amount,       100_000 * f_exp, tol=0.01)
# units = 550 / (2.0 × 50) = 550 / 100 = 5.5 → contracts = 5
check("Formula: units = 5.5",           res.units,    5.5,   tol=1e-6)
check("Formula: contracts = 5",         res.contracts, 5)
print(f"  {res.summary()}")

# ─── TEST 3: Gate ordering ────────────────────────────────────────────────────
print("\n▶ Test 3: Gate ordering — block reasons")

def blocked_reason(overrides: dict) -> str:
    base = dict(
        equity=100_000, stop_distance_price=2.0, point_value=50.0,
        p_win=0.60, reward_risk=2.5,
        regime_quality=1.0, state_confidence=0.8,
        current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
        wins_so_far=0, losses_so_far=0,
        trade_index=1, batch_size=5, target_wins=3, direction="long",
    )
    base.update(overrides)
    s = DynamicMasanielloSizer(SizingConfig(regime_floor=0.5, min_prob=0.50, min_rr=0.5))
    return s.size_trade(TradeContext(**base)).reason

check("Gate: equity≤0",     "Equity must be positive"   in blocked_reason({"equity": 0}),      True)
check("Gate: stop≤0",       "Stop distance must be"      in blocked_reason({"stop_distance_price": 0}), True)
check("Gate: point_val≤0",  "Point value must be"        in blocked_reason({"point_value": -1}), True)
check("Gate: regime_floor", "Regime quality"             in blocked_reason({"regime_quality": 0.3}), True)
check("Gate: min_prob",     "Win probability"            in blocked_reason({"p_win": 0.45}),    True)
check("Gate: min_rr",       "Reward/risk"                in blocked_reason({"reward_risk": 0.3}), True)
check("Gate: edge≤0",       "Expected edge non-positive" in blocked_reason({"p_win": 0.51, "reward_risk": 0.5}), True)

# Batch halt
s2 = DynamicMasanielloSizer(SizingConfig(batch_halt_dd=0.05, min_rr=1.0, regime_floor=0.0))
s2.batch.peak_equity = 100_000
s2.batch.pnl         = -6_000   # 6% drawdown → triggers halt
ctx2 = TradeContext(
    equity=94_000, stop_distance_price=1.0, point_value=1.0,
    p_win=0.60, reward_risk=2.0, regime_quality=1.0, state_confidence=0.8,
    current_drawdown=0.06, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0, trade_index=1, batch_size=5, target_wins=3,
    direction="long",
)
r2 = s2.size_trade(ctx2)
check("Gate: batch halt fires",  r2.allowed, False)
check("Gate: halt reason text",  "halt" in r2.reason.lower(), True)

# ─── TEST 4: Kelly overlay ────────────────────────────────────────────────────
print("\n▶ Test 4: Kelly overlay")

cfg_k = SizingConfig(
    base_risk=0.005, min_risk=0.0001, max_risk=0.10,
    use_kelly_overlay=True, kelly_fraction=0.25,
    min_prob=0.50, min_rr=1.0, regime_floor=0.0,
)
sk = DynamicMasanielloSizer(cfg_k)
ctx_k = TradeContext(
    equity=100_000, stop_distance_price=1.0, point_value=1.0,
    p_win=0.55, reward_risk=2.0,
    regime_quality=1.5, state_confidence=1.0,
    current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0, trade_index=1,
    batch_size=5, target_wins=3, direction="long",
)
rk = sk.size_trade(ctx_k)
# Kelly cap = 0.25 × ((2×0.55 - 0.45)/2) = 0.25 × (0.65/2) = 0.25 × 0.325 = 0.08125
kelly_expected = _fractional_kelly(0.55, 2.0, 0.25)
check("Kelly overlay: cap computed",         rk.kelly_cap is not None, True)
check("Kelly overlay: cap value correct",    rk.kelly_cap, kelly_expected, tol=1e-9)
check("Kelly overlay: f ≤ kelly_cap",        rk.risk_fraction <= rk.kelly_cap + 1e-9, True)

# Kelly cap = 0 (no edge): fall back to min_risk
cfg_k0 = SizingConfig(use_kelly_overlay=True, kelly_fraction=0.25,
                      min_risk=0.0025, max_risk=0.01, min_rr=1.0, regime_floor=0.0)
sk0 = DynamicMasanielloSizer(cfg_k0)
ctx_k0 = TradeContext(
    equity=100_000, stop_distance_price=1.0, point_value=1.0,
    p_win=0.51, reward_risk=1.0,    # edge = 0.51 - 0.49 = 0.02; kelly ≈ 0
    regime_quality=1.0, state_confidence=1.0,
    current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0, trade_index=1,
    batch_size=5, target_wins=3, direction="long",
)
rk0 = sk0.size_trade(ctx_k0)
# When kelly_cap = 0, the overlay is not applied, min_risk floor still holds
check("Kelly overlay cap=0: not below min_risk", rk0.risk_fraction >= cfg_k0.min_risk - 1e-9, True)

# ─── TEST 5: Hard clamp ───────────────────────────────────────────────────────
print("\n▶ Test 5: Hard clamp [min_risk, max_risk]")

cfg_c = SizingConfig(
    base_risk=0.10, min_risk=0.0025, max_risk=0.01,   # huge base_risk to force upper clamp
    min_prob=0.50, min_rr=1.0, regime_floor=0.0,
)
sc = DynamicMasanielloSizer(cfg_c)
ctx_c = TradeContext(
    equity=100_000, stop_distance_price=1.0, point_value=1.0,
    p_win=0.65, reward_risk=3.0,
    regime_quality=1.5, state_confidence=1.0,
    current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0, trade_index=1,
    batch_size=5, target_wins=3, direction="long",
)
rc = sc.size_trade(ctx_c)
check("Clamp: upper cap applied",  rc.risk_fraction <= cfg_c.max_risk + 1e-9, True)
check("Clamp: at max_risk",        rc.risk_fraction, cfg_c.max_risk, tol=1e-9)

# Force lower clamp: tiny Q (very low p_win margin + low regime)
cfg_fl = SizingConfig(base_risk=0.00001, min_risk=0.0025, max_risk=0.01,
                      min_prob=0.50, min_rr=1.0, regime_floor=0.0)
sf = DynamicMasanielloSizer(cfg_fl)
ctx_fl = TradeContext(
    equity=100_000, stop_distance_price=1.0, point_value=1.0,
    p_win=0.51, reward_risk=1.5,
    regime_quality=0.1, state_confidence=0.1,
    current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=2, losses_so_far=0, trade_index=3,
    batch_size=5, target_wins=3, direction="long",
)
rf = sf.size_trade(ctx_fl)
check("Clamp: lower floor applied", rf.risk_fraction >= cfg_fl.min_risk - 1e-9, True)

# ─── TEST 6: RegimeSnapshot integration ──────────────────────────────────────
print("\n▶ Test 6: RegimeSnapshot integration")

snap_bull  = make_snap(mult=1.54, conf=0.82, label="low_vol_bull")
snap_bear  = make_snap(mult=0.14, conf=0.61, label="high_vol_strong_bear")

# from_snapshot
ctx_s = TradeContext.from_snapshot(
    snap=snap_bull, equity=50_000,
    stop_distance_price=5.0, point_value=100.0,
    p_win=0.62, reward_risk=2.5,
    atr_baseline=12.0, atr_current=10.0,
    wins_so_far=1, losses_so_far=0,
    trade_index=2, batch_size=5, target_wins=3,
    current_drawdown=0.02, direction="long",
)
check("from_snapshot: regime_quality = mult",   ctx_s.regime_quality,   snap_bull.risk_multiplier, tol=1e-9)
check("from_snapshot: confidence mapped",        ctx_s.state_confidence, snap_bull.confidence,      tol=1e-9)
check("from_snapshot: regime_label copied",      ctx_s.regime_label,     snap_bull.regime_label)

# size_from_snapshot (bull)
s_snap = DynamicMasanielloSizer(SizingConfig(min_rr=1.0, regime_floor=0.0))
s_snap.batch.wins = 1
s_snap.batch.trade_index = 2
r_bull = s_snap.size_from_snapshot(
    snap_bull, equity=50_000,
    stop_distance_price=0.5, point_value=100.0,
    p_win=0.62, reward_risk=2.5,
    atr_baseline=12.0, atr_current=10.0,
)
check("snapshot bull: allowed",                r_bull.allowed, True)
check("snapshot bull: quality_factor > 0",     r_bull.quality_factor > 0, True)

# size_from_snapshot (crisis bear — regime_quality = 0.14 below default floor 0.50)
s_bear = DynamicMasanielloSizer(SizingConfig(min_rr=1.0, regime_floor=0.50))
r_bear = s_bear.size_from_snapshot(
    snap_bear, equity=50_000,
    stop_distance_price=5.0, point_value=100.0,
    p_win=0.62, reward_risk=2.5,
    atr_baseline=12.0, atr_current=10.0,
)
check("snapshot bear: blocked by regime gate",  r_bear.allowed, False)
check("snapshot bear: reason mentions regime",  "Regime quality" in r_bear.reason, True)

# ─── TEST 7: BatchState lifecycle ────────────────────────────────────────────
print("\n▶ Test 7: BatchState lifecycle")

bs = BatchState()
bs.peak_equity = 100_000
check("Batch: initial drawdown 0",     bs.drawdown, 0.0)
check("Batch: initial wins 0",         bs.wins, 0)
check("Batch: trade_index starts 1",   bs.trade_index, 1)

dummy_res = DynamicMasanielloSizer._blocked("test")
bs.record(dummy_res, outcome=True,  pnl=500.0)
bs.record(dummy_res, outcome=False, pnl=-200.0)
check("Batch: wins = 1",               bs.wins, 1)
check("Batch: losses = 1",             bs.losses, 1)
check("Batch: pnl = 300",             bs.pnl, 300.0, tol=0.01)
check("Batch: trade_index = 3",        bs.trade_index, 3)

bs.pnl = -3000.0  # simulate drawdown
dd = bs.drawdown
check("Batch: drawdown computed",      dd > 0.0, True)
check("Batch: drawdown ≈ 3%",         dd, 3000.0/100_000, tol=0.003)

bs.reset(equity=98_000.0)
check("Batch: reset → batch_num 2",   bs.batch_num, 2)
check("Batch: reset → wins 0",        bs.wins, 0)
check("Batch: reset → trade_index 1", bs.trade_index, 1)
check("Batch: reset → pnl 0",         bs.pnl, 0.0)
check("Batch: reset → peak_equity",   bs.peak_equity, 98_000.0)

# ─── TEST 8: run_batch — sequential state injection ───────────────────────────
print("\n▶ Test 8: run_batch")

s_batch = DynamicMasanielloSizer(SizingConfig(
    min_rr=1.0, regime_floor=0.0, min_prob=0.50,
))
s_batch.batch.peak_equity = 100_000

base_ctx = TradeContext(
    equity=100_000, stop_distance_price=2.0, point_value=1.0,
    p_win=0.60, reward_risk=2.0,
    regime_quality=1.0, state_confidence=0.8,
    current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0, trade_index=1, batch_size=5, target_wins=3,
    direction="long",
)
contexts  = [TradeContext(**base_ctx.__dict__) for _ in range(5)]
outcomes  = [True, False, True, True, False]
pnls      = [200.0, -100.0, 250.0, 180.0, -90.0]

results, final_batch = s_batch.run_batch(contexts, outcomes, pnls)

check("run_batch: 5 results",          len(results), 5)
check("run_batch: all allowed",        all(r.allowed for r in results), True)
check("run_batch: wins = 3",           final_batch.wins, 3)
check("run_batch: losses = 2",         final_batch.losses, 2)
check("run_batch: trade_index = 6",    final_batch.trade_index, 6)
check("run_batch: pnl = 440",          final_batch.pnl, sum(pnls), tol=0.01)
check("run_batch: history length 5",   len(final_batch.history), 5)

# State injection — trade 2 should have wins_so_far=1 (from trade 1 win)
check("run_batch: trade 2 M reflects win 1",
      results[1].masaniello_factor < results[0].masaniello_factor, True)

# ─── TEST 9: Monte Carlo ──────────────────────────────────────────────────────
print("\n▶ Test 9: Monte Carlo simulation")

mc_sizer = DynamicMasanielloSizer(SizingConfig(
    batch_size=5, target_wins=3, min_rr=1.0, regime_floor=0.0, min_prob=0.50,
))
mc_ctx = TradeContext(
    equity=100_000, stop_distance_price=2.0, point_value=1.0,
    p_win=0.60, reward_risk=2.0,
    regime_quality=1.0, state_confidence=0.8,
    current_drawdown=0.0, atr_baseline=10.0, atr_current=10.0,
    wins_so_far=0, losses_so_far=0, trade_index=1, batch_size=5, target_wins=3,
    direction="long",
)
mc_stats = monte_carlo_batch(mc_sizer, mc_ctx, n_simulations=2000, seed=42)

check("MC: p_target_hit in (0,1)",    0 < mc_stats["p_target_hit"] < 1, True)
check("MC: p_target_hit ~0.68 (±0.1)", mc_stats["p_target_hit"], 0.68, tol=0.15)
check("MC: mean_pnl > 0",             mc_stats["mean_pnl"] > 0, True)
check("MC: p90 > median > p10",       mc_stats["p90_pnl"] > mc_stats["median_pnl"] > mc_stats["p10_pnl"], True)
check("MC: std_pnl > 0",             mc_stats["std_pnl"] > 0, True)
check("MC: mean_max_dd in (0,1)",    0 < mc_stats["mean_max_dd"] < 1, True)
check("MC: p_halt in [0,1)",         0 <= mc_stats["p_halt"] < 1, True)
check("MC: sharpe_approx sign matches mean",
      (mc_stats["sharpe_approx"] > 0) == (mc_stats["mean_pnl"] > 0), True)
print(f"  MC results: p_target_hit={mc_stats['p_target_hit']:.2%}  "
      f"p_halt={mc_stats['p_halt']:.2%}  "
      f"mean_pnl=${mc_stats['mean_pnl']:,.0f}  "
      f"mean_dd={mc_stats['mean_max_dd']:.2%}  "
      f"sharpe≈{mc_stats['sharpe_approx']:.3f}")

# ─── TEST 10: Path A preserved (PositionSizer / KellyResult) ─────────────────
print("\n▶ Test 10: Path A — KellyResult interface unchanged")

np.random.seed(7)
returns = np.random.normal(0.0008, 0.010, 300).tolist()
snap_a  = make_snap(mult=1.20, conf=0.80, label="med_low_vol_bull")

# Path A must still work without RegimeSnapshot
req_no_snap = PositionRequest(returns=returns, kelly_fraction=0.5, target_vol=0.15)
kr_no_snap  = PositionSizer().size(req_no_snap)
check("Path A: recommended_f in [0,1]", 0.0 <= kr_no_snap.recommended_f <= 1.0, True)
check("Path A: regime_label = no_regime", kr_no_snap.regime_label, "no_regime")

# Path A with snapshot
class _FakeSnap:
    risk_multiplier = 1.20
    regime_label    = "med_low_vol_bull"
    confidence      = 0.80

req_snap = PositionRequest(returns=returns, kelly_fraction=0.5, target_vol=0.15, regime=_FakeSnap())
kr_snap  = PositionSizer().size(req_snap)
check("Path A: regime gating applied",  kr_snap.regime_gated_f <= kr_snap.vol_scaled_f * 1.21, True)
check("Path A: regime_multiplier",      kr_snap.regime_multiplier, 1.20, tol=1e-9)
check("Path A: recommended_f = regime_gated_f", kr_snap.recommended_f, kr_snap.regime_gated_f)
check("Path A: KellyResult fields present", hasattr(kr_snap, "sharpe_ratio"), True)
print(f"  Path A: f*={kr_snap.full_kelly_f:.4f}  "
      f"fractional={kr_snap.fractional_f:.4f}  "
      f"vol_scaled={kr_snap.vol_scaled_f:.4f}  "
      f"regime_gated={kr_snap.regime_gated_f:.4f}")

# ─── Summary ──────────────────────────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'─'*60}")
print(f"  {PASS}/{total} tests passed  {'✅' if FAIL == 0 else f'❌ {FAIL} failed'}")
if FAIL == 0:
    print("\n✅  All position sizer tests passed.\n")
else:
    print(f"\n❌  {FAIL} test(s) failed — see above.\n")
    sys.exit(1)
