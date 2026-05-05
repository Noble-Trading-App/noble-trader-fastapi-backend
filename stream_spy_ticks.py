#!/usr/bin/env python3
"""
SPY Tick Stream Tester
══════════════════════
Seeds SPY with 150 bars of synthetic history, then pushes 50 live ticks
one-by-one via POST /stream/tick and prints a formatted TickResponse for each.

Demonstrates:
  • Seeding a session (POST /stream/seed)
  • Ingesting live ticks (POST /stream/tick)
  • Parsing and pretty-printing TickResponse
  • Detecting regime changes mid-stream
  • Printing a summary after all ticks

Usage
─────
  python stream_spy_ticks.py                    # localhost:8000, default settings
  python stream_spy_ticks.py --url http://x:8000
  python stream_spy_ticks.py --ticks 100 --interval 0.2
  python stream_spy_ticks.py --regime crisis    # seed with crisis-like prices

Options
  --url       Base URL of the server  (default: http://localhost:8000)
  --symbol    Symbol to seed/stream   (default: SPY)
  --seed-bars Number of historical bars to seed with (default: 150, min 81)
  --ticks     Number of live ticks to push (default: 50)
  --interval  Seconds between ticks   (default: 0.1)
  --regime    Seed regime: bull | bear | crisis | mixed (default: mixed)
  --mu        Manual return drift for live ticks (overrides --regime live phase)
  --sigma     Manual return vol for live ticks
  --no-color  Disable ANSI colours (useful for log files)
  --json      Print raw JSON instead of formatted output
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ── Colour helpers ────────────────────────────────────────────────────────────

def _supports_colour(no_color: bool) -> bool:
    return not no_color and sys.stdout.isatty()


class C:
    """ANSI colour codes — disabled when NO_COLOR=True."""
    _on = True

    @classmethod
    def enable(cls, on: bool):
        cls._on = on

    @classmethod
    def _w(cls, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if cls._on else s

    @classmethod
    def bold(cls, s):    return cls._w("1",    s)
    @classmethod
    def dim(cls, s):     return cls._w("2",    s)
    @classmethod
    def green(cls, s):   return cls._w("32",   s)
    @classmethod
    def red(cls, s):     return cls._w("31",   s)
    @classmethod
    def yellow(cls, s):  return cls._w("33",   s)
    @classmethod
    def cyan(cls, s):    return cls._w("36",   s)
    @classmethod
    def magenta(cls, s): return cls._w("35",   s)
    @classmethod
    def blue(cls, s):    return cls._w("34",   s)
    @classmethod
    def grey(cls, s):    return cls._w("90",   s)


# ── Synthetic price generation ────────────────────────────────────────────────

REGIME_PARAMS = {
    "bull":   {"seed_mu": 0.0010, "seed_sigma": 0.006, "live_mu":  0.0008, "live_sigma": 0.007},
    "bear":   {"seed_mu":-0.0008, "seed_sigma": 0.018, "live_mu": -0.0010, "live_sigma": 0.020},
    "crisis": {"seed_mu":-0.0018, "seed_sigma": 0.030, "live_mu": -0.0020, "live_sigma": 0.035},
    "mixed":  {"seed_mu": 0.0002, "seed_sigma": 0.012, "live_mu": -0.0005, "live_sigma": 0.022},
}


def generate_prices(n: int, mu: float, sigma: float, start: float = 100.0, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n)
    prices = start * np.cumprod(1 + r)
    return [round(float(p), 4) for p in prices]


def generate_live_ticks(n: int, start: float, mu: float, sigma: float, seed: int = 99) -> list[float]:
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n)
    prices = start * np.cumprod(1 + r)
    return [round(float(p), 4) for p in prices]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def post(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} — {body[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}. Is the server running?")


def get(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}")


# ── Regime colour mapping ─────────────────────────────────────────────────────

def regime_colour(label: str) -> str:
    if "strong_bear" in label or ("high" in label and "bear" in label):
        return C.red(label)
    if "bear" in label or "high" in label:
        return C.yellow(label)
    if "strong_bull" in label or ("low" in label and "bull" in label):
        return C.green(label)
    if "bull" in label:
        return C.cyan(label)
    return C.grey(label)


def mult_colour(mult: float) -> str:
    s = f"{mult:.3f}×"
    if mult >= 1.20:  return C.green(s)
    if mult >= 0.70:  return C.yellow(s)
    return C.red(s)


def prob_bar(val: float, width: int = 12) -> str:
    filled = int(round(val * width))
    bar    = "█" * filled + "░" * (width - filled)
    return bar


# ── Tick printer ──────────────────────────────────────────────────────────────

def print_tick(tick: dict, tick_num: int, prev_regime: Optional[str], raw_json: bool):
    if raw_json:
        print(json.dumps(tick, indent=2))
        return

    regime_changed = tick.get("regime_changed", False)
    regime_label   = tick.get("regime_label", "—")
    alert          = tick.get("alert")

    # Separator + regime-change banner
    if tick_num == 1 or regime_changed:
        print()
        if regime_changed and prev_regime:
            print(C.bold(C.magenta(f"  ⚡ REGIME CHANGE  {prev_regime}  →  {regime_label}")))
            if alert:
                print(C.dim(f"     {alert}"))
        print(C.dim("  " + "─" * 72))

    ts_str  = datetime.fromtimestamp(tick["ts"], tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    changed = C.magenta(" ⚡CHANGED") if regime_changed else ""

    # Main line
    price_str  = f"{tick['price']:>8.3f}"
    tick_label = f"#{tick_num:02d}"
    print(
        f"  {C.bold(C.grey(tick_label))}  "
        f"{C.grey(ts_str)}  "
        f"${C.bold(price_str)}  "
        f"bars={C.grey(str(tick['n_bars']))}  "
        f"regime={regime_colour(regime_label)}{changed}"
    )

    # Factor line
    rec_f   = tick.get("recommended_f", 0)
    mult    = tick.get("risk_multiplier", 1.0)
    conf    = tick.get("confidence", 0.0)
    sharpe  = tick.get("sharpe_ratio", 0.0)
    var95   = tick.get("var_95", 0.0)
    stop    = tick.get("suggested_stop", 0.0)
    tp      = tick.get("suggested_tp", 0.0)
    refits  = tick.get("refit_count", 0)

    print(
        f"       "
        f"f*={C.cyan(f'{rec_f:.4f}')}  "
        f"mult={mult_colour(mult)}  "
        f"conf={C.grey(f'{conf:.2f}')}  "
        f"sharpe={C.grey(f'{sharpe:+.2f}')}  "
        f"VaR95={C.yellow(f'{var95:.4f}')}  "
        f"stop={C.red(f'{stop:.4f}')}  "
        f"tp={C.green(f'{tp:.4f}')}  "
        f"refits={C.grey(str(refits))}"
    )

    # Vol and trend prob bars (compact)
    vp = tick.get("vol_probs", {})
    tp2 = tick.get("trend_probs", {})
    vol_bars   = "  ".join(f"{k[:3]}={prob_bar(v,8)} {v:.2f}" for k, v in vp.items())
    trend_bars = "  ".join(f"{k[:3]}={prob_bar(v,8)} {v:.2f}" for k, v in tp2.items())
    print(f"       {C.dim('vol  ')} {C.grey(vol_bars)}")
    print(f"       {C.dim('trend')} {C.grey(trend_bars)}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(symbol: str, ticks: list[dict], latencies: list[float],
                  seed_bars: int, raw_json: bool):
    if raw_json:
        return

    print()
    print(C.bold(C.cyan("  ══ Stream Summary ══")))
    print(f"  Symbol       {symbol}")
    print(f"  Seed bars    {seed_bars}")
    print(f"  Ticks pushed {len(ticks)}")

    regimes = [t["regime_label"] for t in ticks]
    unique  = list(dict.fromkeys(regimes))  # preserve order, deduplicate
    changes = sum(1 for t in ticks if t.get("regime_changed"))
    print(f"  Regimes seen {len(set(regimes))}  →  {' → '.join(regime_colour(r) for r in unique)}")
    print(f"  Regime changes {changes}")

    rec_fs = [t["recommended_f"] for t in ticks]
    mults  = [t["risk_multiplier"] for t in ticks]
    print(f"  recommended_f   min={min(rec_fs):.4f}  max={max(rec_fs):.4f}  final={rec_fs[-1]:.4f}")
    print(f"  risk_multiplier min={min(mults):.3f}×  max={max(mults):.3f}×  final={mults[-1]:.3f}×")

    vars95 = [t["var_95"] for t in ticks]
    print(f"  VaR 95%        min={min(vars95):.4f}  max={max(vars95):.4f}")

    if latencies:
        mean_ms = sum(latencies) / len(latencies) * 1000
        max_ms  = max(latencies) * 1000
        print(f"  Latency (RTT)   mean={mean_ms:.1f}ms  max={max_ms:.1f}ms  n={len(latencies)}")

    final = ticks[-1]
    rec_f_str = f"{final['recommended_f']:.4f}"
    var_str   = f"{final['var_95']:.4f}"
    stop_str  = f"{final['suggested_stop']:.4f}"
    tp_str    = f"{final['suggested_tp']:.4f}"
    print("\n  Final tick:")
    print(f"    regime        {regime_colour(final['regime_label'])}")
    print(f"    recommended_f {C.cyan(rec_f_str)}")
    print(f"    VaR 95%       {C.yellow(var_str)}")
    print(f"    suggested_stop {C.red(stop_str)}")
    print(f"    suggested_tp   {C.green(tp_str)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed SPY and push 50 live ticks via POST /stream/tick",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url",       default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--symbol",    default="SPY",                   help="Symbol")
    parser.add_argument("--seed-bars", default=150,    type=int,        help="Historical bars to seed (≥81)")
    parser.add_argument("--ticks",     default=50,     type=int,        help="Live ticks to push")
    parser.add_argument("--interval",  default=0.1,    type=float,      help="Seconds between ticks")
    parser.add_argument("--regime",    default="mixed",
                        choices=["bull", "bear", "crisis", "mixed"],    help="Seed price regime")
    parser.add_argument("--mu",        default=None,   type=float,      help="Override live tick drift")
    parser.add_argument("--sigma",     default=None,   type=float,      help="Override live tick vol")
    parser.add_argument("--no-color",  action="store_true",             help="Disable ANSI colours")
    parser.add_argument("--json",      action="store_true",             help="Print raw JSON per tick")
    args = parser.parse_args()

    C.enable(_supports_colour(args.no_color))

    params    = REGIME_PARAMS[args.regime]
    live_mu   = args.mu    if args.mu    is not None else params["live_mu"]
    live_sigma = args.sigma if args.sigma is not None else params["live_sigma"]
    base_url  = args.url.rstrip("/")

    if not args.json:
        print()
        print(C.bold(f"  Regime Risk Platform — SPY Tick Stream Test"))
        print(C.dim(f"  Server: {base_url}"))
        print(C.dim(f"  Symbol: {args.symbol} | Seed: {args.seed_bars} bars | "
                    f"Ticks: {args.ticks} | Regime: {args.regime}"))

    # ── 1. Health check ───────────────────────────────────────────────────────
    if not args.json:
        print(f"\n  {C.bold('Step 1')} — Health check")
    try:
        health = get(f"{base_url}/health")
        if not args.json:
            print(f"  {C.green('✓')}  Server OK — v{health.get('version','?')}  "
                  f"sessions={health.get('sessions',0)}  "
                  f"redis={health.get('redis',False)}")
    except RuntimeError as e:
        print(f"\n  {C.red('✗')}  {e}\n")
        sys.exit(1)

    # ── 2. Generate seed prices ───────────────────────────────────────────────
    if not args.json:
        print(f"\n  {C.bold('Step 2')} — Generating {args.seed_bars} seed bars ({args.regime} regime)")

    seed_prices = generate_prices(
        args.seed_bars, params["seed_mu"], params["seed_sigma"], start=100.0, seed=42
    )
    if not args.json:
        print(f"  {C.grey(f'Price range: {min(seed_prices):.2f} – {max(seed_prices):.2f}')}")

    # ── 3. Seed the session ───────────────────────────────────────────────────
    if not args.json:
        print(f"\n  {C.bold('Step 3')} — Seeding {args.symbol} session ...")

    seed_payload = {
        "symbol":       args.symbol,
        "prices":       seed_prices,
        "window":       500,
        "kelly_fraction": 0.5,
        "target_vol":   0.15,
        "base_risk_limit": 0.02,
        "refit_every":  50,
    }

    try:
        status = post(f"{base_url}/stream/seed", seed_payload)
    except RuntimeError as e:
        print(f"\n  {C.red('✗')}  Seed failed: {e}\n")
        sys.exit(1)

    if not args.json:
        ready = status.get("ready", False)
        icon  = C.green("✓") if ready else C.yellow("⚠")
        print(f"  {icon}  Session seeded — ready={ready}  n_bars={status.get('n_bars',0)}  "
              f"refits={status.get('refit_count',0)}  "
              f"regime={status.get('last_regime','pending')}")
        if not ready:
            print(f"  {C.yellow('⚠')}  Session not ready — add more seed bars (--seed-bars 200)")

    # ── 4. Generate live ticks ────────────────────────────────────────────────
    live_prices = generate_live_ticks(
        args.ticks,
        start=seed_prices[-1],
        mu=live_mu,
        sigma=live_sigma,
        seed=99,
    )

    if not args.json:
        print(f"\n  {C.bold('Step 4')} — Pushing {args.ticks} live ticks "
              f"(μ={live_mu:+.4f}, σ={live_sigma:.4f})")
        print(f"  {C.dim('(live price range preview: '  )}"
              f"{C.grey(f'{min(live_prices):.2f} – {max(live_prices):.2f})')}")

    # ── 5. Push ticks ─────────────────────────────────────────────────────────
    all_ticks:   list[dict] = []
    latencies:   list[float] = []
    prev_regime: Optional[str] = status.get("last_regime")

    for i, price in enumerate(live_prices, start=1):
        tick_payload = {
            "symbol": args.symbol,
            "price":  price,
            "ts":     time.time(),
        }

        t0 = time.perf_counter()
        try:
            result = post(f"{base_url}/stream/tick", tick_payload)
        except RuntimeError as e:
            print(f"\n  {C.red('✗')}  Tick {i} failed: {e}")
            break
        latency = time.perf_counter() - t0
        latencies.append(latency)

        # Handle buffering stub (session not yet warm)
        if result.get("status") == "buffering":
            if not args.json:
                print(f"  {C.grey(f'#{i:02d}')}  {C.dim('buffering ...')}  "
                      f"n_bars={result.get('n_bars', 0)}")
            if args.interval > 0:
                time.sleep(args.interval)
            continue

        all_ticks.append(result)
        print_tick(result, i, prev_regime, raw_json=args.json)
        prev_regime = result.get("regime_label", prev_regime)

        if args.interval > 0:
            time.sleep(args.interval)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    if all_ticks:
        print_summary(args.symbol, all_ticks, latencies, args.seed_bars, args.json)
    else:
        print(f"\n  {C.yellow('⚠')}  No TickResponse received — "
              "session may still be warming up. Try --seed-bars 200.\n")


if __name__ == "__main__":
    main()
