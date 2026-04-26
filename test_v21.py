"""
v2.1 Feature Tests — Simulation · Portfolio · Redis · Auth
Run: python test_v21.py
"""
import sys, warnings
sys.path.insert(0, "/home/claude/v21")
# Remove bare /home/claude to avoid rp4 package shadowing regime_platform
sys.path = [p for p in sys.path if p not in ("/home/claude", "")]
warnings.filterwarnings("ignore")

import numpy as np
import asyncio

# ── Synthetic data ────────────────────────────────────────────────────────────
np.random.seed(42)

def make_prices(n, mu, sigma, start=100.0):
    r = np.random.normal(mu, sigma, n)
    return list(start * np.cumprod(1 + r))

prices_spy = make_prices(300, 0.0008, 0.010, 100.0)
prices_qqq = make_prices(300,-0.0005, 0.018, 200.0)
prices_gld = make_prices(300, 0.0002, 0.006, 180.0)

print("=" * 64)
print("  v2.1 Feature Tests")
print("=" * 64)

# ─── TEST 1: Regime Simulator ─────────────────────────────────────────────────
print("\n▶ Test 1: RegimeSimulator")
from regime_platform.core.simulator import RegimeSimulator

sim    = RegimeSimulator()
prices = np.array(prices_spy)
result = sim.simulate(prices, symbol="SPY", horizon=20, n_paths=200, seed=42)

assert result.horizon  == 20
assert result.n_paths  == 200
assert len(result.price_median)       == 20
assert len(result.price_p5)           == 20
assert len(result.expected_risk_mult) == 20
assert len(result.regime_occupancy)   == 20
assert 0.0 <= result.pct_paths_positive <= 1.0
assert result.max_drawdown_mean <= 0.0
assert result.return_var95 >= 0.0
assert result.terminal_regime_mode in sim._all_regimes

# Price fan ordering
for i in range(20):
    assert result.price_p5[i] <= result.price_p25[i] <= result.price_median[i]
    assert result.price_median[i] <= result.price_p75[i] <= result.price_p95[i]

# Regime occupancy sums to 1 per step
for step_occ in result.regime_occupancy:
    total = sum(step_occ.values())
    assert abs(total - 1.0) < 1e-4, f"Occupancy sum = {total}"

print(f"  Current regime:     {result.current_regime}")
print(f"  Terminal mode:      {result.terminal_regime_mode}")
print(f"  Return mean/std:    {result.return_mean:.4f} / {result.return_std:.4f}")
print(f"  VaR95:              {result.return_var95:.4f}")
print(f"  % positive paths:   {result.pct_paths_positive:.2%}")
print(f"  Mean max drawdown:  {result.max_drawdown_mean:.4f}")
print(f"  Price p5→p95 at T20: {result.price_p5[-1]:.2f} → {result.price_p95[-1]:.2f}")
print("  ✓ All assertions passed")

# ─── TEST 2: All 16 combined regimes reachable ────────────────────────────────
print("\n▶ Test 2: 16-state combined system")
all_regimes = sim._all_regimes
assert len(all_regimes) == 16, f"Expected 16 combined regimes, got {len(all_regimes)}"

# Verify Kronecker product produces 16×16 matrix
from regime_platform.core.regime_engine import RegimeHMM
model = RegimeHMM()
model.fit(prices)
returns = np.diff(np.log(prices))
T16, regime_stats = sim._build_combined_system(model, returns)
assert T16.shape == (16, 16), f"Expected (16,16), got {T16.shape}"
row_sums = T16.sum(axis=1)
assert np.all(np.abs(row_sums - 1.0) < 1e-4), "T16 rows don't sum to 1"
print(f"  Combined regimes:   {len(all_regimes)}")
print(f"  Transition matrix:  {T16.shape}")
print(f"  Row sum range:      [{row_sums.min():.4f}, {row_sums.max():.4f}]")
print("  ✓ 16-state Kronecker system valid")

# ─── TEST 3: Portfolio Service ────────────────────────────────────────────────
print("\n▶ Test 3: PortfolioService")

async def test_portfolio():
    from regime_platform.services.portfolio_service import PortfolioService
    from regime_platform.services.registry import registry

    svc = PortfolioService()

    # Seed 3 sessions
    for sym, prices_list in [("SPY", prices_spy), ("QQQ", prices_qqq), ("GLD", prices_gld)]:
        session = await registry.get_or_create(symbol=sym, window=300)
        await session.seed(prices_list)

    result = await svc.summarise()

    assert result.n_symbols == 3
    assert len(result.symbols) == 3
    assert all(s.ready for s in result.symbols)
    assert result.dominant_regime != "pending"
    assert 0.0 <= result.regime_consensus <= 1.0
    assert result.avg_risk_multiplier > 0.0
    assert result.portfolio_var95 >= 0.0

    print(f"  Symbols:            {[s.symbol for s in result.symbols]}")
    print(f"  Dominant regime:    {result.dominant_regime} ({result.regime_consensus:.0%} consensus)")
    print(f"  Avg risk mult:      {result.avg_risk_multiplier:.4f}×")
    print(f"  Avg f*:             {result.avg_recommended_f:.4f}")
    print(f"  Portfolio VaR95:    {result.portfolio_var95:.4f}")
    print(f"  High-risk count:    {result.high_risk_count}")
    print(f"  Concentration flag: {result.concentration_flag}")
    print(f"  Divergence flag:    {result.regime_divergence_flag}")
    print("  ✓ Portfolio aggregation passed")

asyncio.run(test_portfolio())

# ─── TEST 4: Redis Persistence (no-op without REDIS_URL) ─────────────────────
print("\n▶ Test 4: RedisPersistence (no-op mode)")
from regime_platform.services.redis_persistence import RedisPersistence

p = RedisPersistence(redis_url="")
assert p.enabled == False
print(f"  Enabled:            {p.enabled}")

async def test_redis_noop():
    # All methods should be no-ops
    n = await p.restore(None)
    assert n == 0
    await p.append_price("TEST", 100.0)  # should not raise
    meta = await p.load_meta("TEST")
    assert meta == {}
    syms = await p.list_persisted_symbols()
    assert syms == []
    print("  ✓ No-op mode: all methods return safely without REDIS_URL")

asyncio.run(test_redis_noop())

# ─── TEST 5: JWT Auth ─────────────────────────────────────────────────────────
print("\n▶ Test 5: JWT Authentication")
import os
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-unit-tests"
os.environ["AUTH_ENABLED"]   = "true"

# Re-import to pick up env vars
import importlib
import regime_platform.auth.jwt_auth as jwt_mod
importlib.reload(jwt_mod)

# Monkey-patch the module-level vars
jwt_mod.JWT_SECRET_KEY = "test-secret-key-for-unit-tests"
jwt_mod.AUTH_ENABLED   = True

# Create token
token = jwt_mod.create_access_token({"sub": "test_user", "role": "trader"})
assert isinstance(token, str) and len(token) > 20
print(f"  Token (first 30):   {token[:30]}...")

# Decode token
td = jwt_mod.decode_token(token)
assert td.sub == "test_user"
assert td.role == "trader"
assert td.can_write == True
assert td.is_admin == False
print(f"  Decoded sub:        {td.sub}")
print(f"  Decoded role:       {td.role}")
print(f"  can_write:          {td.can_write}")

# Admin token
admin_tok = jwt_mod.create_access_token({"sub": "admin", "role": "admin"})
admin_td  = jwt_mod.decode_token(admin_tok)
assert admin_td.is_admin == True
print(f"  Admin is_admin:     {admin_td.is_admin}")

# make_login_response
resp = jwt_mod.make_login_response("alice", "viewer")
assert resp["token_type"] == "bearer"
assert resp["sub"] == "alice"
assert resp["role"] == "viewer"
print("  ✓ JWT create / decode / role check passed")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─' * 64}")
print("✅  All v2.1 tests passed.\n")
print("  New endpoints available on main_v3.py:")
print("  POST /simulate/{symbol}  — Monte Carlo regime paths")
print("  GET  /portfolio          — Multi-symbol risk aggregation")
print("  POST /auth/token         — Issue JWT")
print("  GET  /auth/me            — Current user info")
print("  GET  /auth/refresh       — Refresh token")
