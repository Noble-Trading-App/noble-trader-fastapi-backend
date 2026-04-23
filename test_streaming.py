"""
Streaming layer smoke test — no HTTP server needed.
Tests StreamSession directly: seed → tick loop → regime transitions → alerts.
Run: python test_streaming.py
"""

import sys, asyncio, time, numpy as np
sys.path.insert(0, "/home/claude")

from regime_platform_v2.services.stream_session import StreamSession, RegimeAlert
from regime_platform_v2.services.registry import registry

# ── Synthetic multi-regime price series ──────────────────────────────────────
np.random.seed(7)
def make_prices(n, mu, sigma, start=100.0):
    r = np.random.normal(mu, sigma, n)
    return list(start * np.cumprod(1 + r))

seed_prices  = make_prices(200, 0.0008, 0.007)   # low-vol bull warm-up
bear_segment  = make_prices(80, -0.0015, 0.025, seed_prices[-1])
recov_segment = make_prices(80,  0.0005, 0.010, bear_segment[-1])
live_prices   = bear_segment + recov_segment

# ── Alert collector ───────────────────────────────────────────────────────────
alerts_received = []

async def on_alert(alert: RegimeAlert):
    alerts_received.append(alert)
    print(f"  🚨 ALERT [{alert.severity.upper()}] {alert.previous} → {alert.current}")

# ── Test ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 62)
    print("  Streaming Layer Smoke Test")
    print("=" * 62)

    # 1. Create session via registry
    session = await registry.get_or_create(
        symbol="TEST",
        window=500,
        kelly_fraction=0.5,
        target_vol=0.15,
        refit_every=40,
    )
    session.add_alert_callback(on_alert)

    # 2. Seed
    print(f"\n[1] Seeding with {len(seed_prices)} historical prices...")
    await session.seed(seed_prices)
    print(f"    ready={session.is_ready}  n_bars={len(session.price_buffer)}  refits={session._refit_count}")
    assert session.is_ready, "Session should be ready after seeding"

    # 3. Subscribe a queue consumer
    q = session.subscribe()
    ticks_received = []

    async def consume():
        while True:
            try:
                tick = await asyncio.wait_for(q.get(), timeout=2.0)
                ticks_received.append(tick)
            except asyncio.TimeoutError:
                break

    consumer_task = asyncio.create_task(consume())

    # 4. Push live ticks
    print(f"\n[2] Streaming {len(live_prices)} live ticks...")
    regime_seq = []
    first_tick_time = time.perf_counter()

    for i, price in enumerate(live_prices):
        tick = await session.tick(price, time.time())
        if tick:
            regime_seq.append(tick.regime_label)
            if i % 30 == 0:
                print(f"    bar {i:3d} | price={price:7.2f} | regime={tick.regime_label:<22} "
                      f"| f*={tick.recommended_f:.4f} | VaR95={tick.var_95:.4f} "
                      f"{'⚡ CHANGED' if tick.regime_changed else ''}")

    elapsed = (time.perf_counter() - first_tick_time) * 1000
    await consumer_task

    # 5. Results
    print(f"\n[3] Results")
    print(f"    Total ticks processed : {session._tick_count}")
    print(f"    HMM refits            : {session._refit_count}")
    print(f"    Regime alerts fired   : {len(alerts_received)}")
    print(f"    Subscriber received   : {len(ticks_received)} ticks")
    print(f"    Total stream time     : {elapsed:.1f}ms  ({elapsed/len(live_prices):.2f}ms/tick)")
    print(f"    Unique regimes seen   : {sorted(set(regime_seq))}")

    # 6. Registry stats
    print(f"\n[4] Registry stats")
    stats = await registry.stats()
    for sym, s in stats.items():
        print(f"    {sym}: {s}")

    # 7. Assertions
    assert session._tick_count > 0
    assert session._refit_count >= 1
    assert len(ticks_received) > 0
    assert len(regime_seq) > 0

    print(f"\n✅  All streaming tests passed.\n")


asyncio.run(main())
