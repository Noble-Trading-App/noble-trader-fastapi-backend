"""
Regime Transition Simulator
════════════════════════════

Simulates future regime and price paths using the HMM transition matrix
extracted from a fitted RegimeHMM.  Two simulation modes:

  1. MarkovChain  — discrete-state Markov chain over combined vol×trend regimes
  2. MonteCarlo   — continuous price paths drawn from per-regime return distributions,
                    with regime switching driven by the Markov chain

Both modes produce `n_paths` forward trajectories of `horizon` steps, yielding:
  - Regime state sequence per path
  - Price trajectory per path
  - Summary statistics: regime occupancy, expected risk multiplier, VaR fan

Design note: this is an endogenous simulation — it does NOT call an external
ODE solver (simupy) because hmmlearn already gives us the discrete-time
transition matrix, which is the natural representation for regime switching.
A continuous-time ODE layer (simupy) would be useful for underlying factor
dynamics, but is out of scope here to avoid hard simupy dependency.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .regime_engine import RegimeHMM, RegimeSnapshot


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class SimPath:
    """Single simulated forward path."""
    regime_labels:    list[str]     # length = horizon
    risk_multipliers: list[float]   # per-step
    price_path:       list[float]   # starting from current_price
    cumulative_return: float        # (price_path[-1] / price_path[0]) - 1


@dataclass
class SimulationResult:
    """Aggregated results across n_paths simulated trajectories."""
    symbol:               str
    horizon:              int
    n_paths:              int
    current_regime:       str
    current_price:        float

    # Per-step regime occupancy probabilities (horizon × n_regimes)
    regime_occupancy:     list[dict[str, float]]

    # Expected risk multiplier at each forward step
    expected_risk_mult:   list[float]

    # Price fan statistics
    price_median:         list[float]
    price_p5:             list[float]
    price_p25:            list[float]
    price_p75:            list[float]
    price_p95:            list[float]

    # Return distribution at horizon end
    return_mean:          float
    return_std:           float
    return_var95:         float    # VaR at 95% confidence (tail loss)
    return_cvar95:        float    # Expected Shortfall

    # Most likely terminal regime
    terminal_regime_mode: str

    # Path-level stats
    pct_paths_positive:   float    # % paths with positive return at horizon
    max_drawdown_mean:    float    # mean of per-path maximum drawdowns


# ─── Simulator ────────────────────────────────────────────────────────────────

class RegimeSimulator:
    """
    Markov-chain regime transition simulator.

    Usage
    ─────
    sim = RegimeSimulator()
    result = sim.simulate(prices, symbol="SPY", horizon=20, n_paths=1000)
    """

    # Combined 4×4 = 16 regime labels in canonical order
    VOL_LABELS   = RegimeHMM.VOL_LABELS     # [low, med_low, med_high, high]
    TREND_LABELS = RegimeHMM.TREND_LABELS   # [strong_bear, bear, bull, strong_bull]

    @property
    def _all_regimes(self) -> list[str]:
        return [
            f"{v}_vol_{t}"
            for v in self.VOL_LABELS
            for t in self.TREND_LABELS
        ]  # 16 combined labels in vol-major order

    def simulate(
        self,
        prices:            np.ndarray,
        symbol:            str = "UNKNOWN",
        horizon:           int = 20,
        n_paths:           int = 500,
        seed:              Optional[int] = 42,
        current_price:     Optional[float] = None,
    ) -> SimulationResult:
        """
        Fit HMM on `prices`, then simulate `n_paths` forward paths of
        `horizon` bars using the fitted transition matrix.

        Parameters
        ──────────
        prices        Close price array (min 81 bars).
        symbol        Label for the result.
        horizon       Number of forward steps to simulate.
        n_paths       Number of independent Monte Carlo paths.
        seed          Random seed for reproducibility.
        current_price Starting price (defaults to prices[-1]).
        """
        rng = np.random.default_rng(seed)

        # Fit HMM
        model = RegimeHMM()
        model.fit(prices)
        snap  = model.predict(prices)

        returns = np.diff(np.log(prices))
        current_price = float(prices[-1]) if current_price is None else current_price

        # Build combined 16-state transition matrix and regime stats
        trans16, regime_stats = self._build_combined_system(model, returns)

        # Find current combined-state index
        current_idx = self._current_state_idx(snap)

        # Run Monte Carlo
        paths = self._run_paths(
            trans16, regime_stats, current_idx, current_price,
            horizon, n_paths, rng
        )

        return self._aggregate(paths, snap, symbol, horizon, n_paths, current_price)

    # ── System construction ───────────────────────────────────────────────────

    def _build_combined_system(
        self,
        model: RegimeHMM,
        returns: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Build a 16×16 combined transition matrix by taking the Kronecker
        product of the vol and trend transition matrices.

        T_combined[i*4+j, k*4+l] = T_vol[i,k] × T_trend[j,l]

        Also compute per-regime (mu, sigma) from historical returns filtered
        by the Viterbi-decoded state sequence.
        """
        T_vol   = model._vol_model.transmat_       # (4,4) — already sanitised
        T_trend = model._trend_model.transmat_     # (4,4)

        # Kronecker product → 16×16
        T16 = np.kron(T_vol, T_trend)

        # Re-normalise rows (floating point drift in kron)
        row_sums = T16.sum(axis=1, keepdims=True)
        T16 = np.where(row_sums > 0, T16 / row_sums, 1.0 / 16)

        # Per-regime return distribution — fit on empirical data
        regime_stats = self._fit_regime_return_stats(model, returns)

        return T16, regime_stats

    def _fit_regime_return_stats(
        self,
        model: RegimeHMM,
        returns: np.ndarray,
    ) -> dict[str, dict]:
        """
        Decode the historical Viterbi path for vol and trend separately,
        then compute (mu, sigma) per combined regime label.
        Falls back to global (mu, sigma) if fewer than 5 obs in a state.
        """
        global_mu    = float(np.mean(returns))
        global_sigma = float(np.std(returns, ddof=1))

        # Viterbi decode
        vol_feats   = model._vol_features(returns)
        trend_feats = model._trend_features(returns)
        vol_states   = model._vol_model.decode(vol_feats)[1]
        trend_states = model._trend_model.decode(trend_feats)[1]

        stats = {}
        for vi, vl in enumerate(self.VOL_LABELS):
            for ti, tl in enumerate(self.TREND_LABELS):
                label = f"{vl}_vol_{tl}"
                # Map HMM raw state index → label index
                raw_vol_idx   = next((k for k, v in model._vol_label_map.items()   if v == vl),   vi)
                raw_trend_idx = next((k for k, v in model._trend_label_map.items() if v == tl), ti)
                mask = (vol_states == raw_vol_idx) & (trend_states == raw_trend_idx)
                r = returns[mask]
                if len(r) >= 5:
                    mu    = float(np.mean(r))
                    sigma = max(float(np.std(r, ddof=1)), 1e-5)
                else:
                    mu, sigma = global_mu, global_sigma
                stats[label] = {"mu": mu, "sigma": sigma}

        return stats

    def _current_state_idx(self, snap: RegimeSnapshot) -> int:
        """Map current regime label to its index in the 16-state system."""
        try:
            return self._all_regimes.index(snap.regime_label)
        except ValueError:
            return 0  # fallback

    # ── Monte Carlo ───────────────────────────────────────────────────────────

    def _run_paths(
        self,
        T16:          np.ndarray,
        regime_stats: dict,
        start_idx:    int,
        start_price:  float,
        horizon:      int,
        n_paths:      int,
        rng:          np.random.Generator,
    ) -> list[SimPath]:
        """Generate n_paths forward trajectories."""
        all_regimes = self._all_regimes
        cum_T = np.cumsum(T16, axis=1)   # for efficient sampling

        paths: list[SimPath] = []

        for _ in range(n_paths):
            state_idx   = start_idx
            price       = start_price
            regime_seq  = []
            mult_seq    = []
            price_seq   = [price]
            peak        = price

            for _ in range(horizon):
                # Transition
                u = rng.random()
                state_idx = int(np.searchsorted(cum_T[state_idx], u))
                state_idx = min(state_idx, 15)

                label = all_regimes[state_idx]
                regime_seq.append(label)

                # Risk multiplier for this state
                snap_tmp = _make_snap(label)
                mult_seq.append(snap_tmp.risk_multiplier)

                # Draw return
                stat  = regime_stats.get(label, {"mu": 0.0, "sigma": 0.01})
                r     = float(rng.normal(stat["mu"], stat["sigma"]))
                price = price * (1.0 + r)
                price = max(price, 1e-4)
                price_seq.append(price)
                peak = max(peak, price)

            cum_ret = (price_seq[-1] / price_seq[0]) - 1.0
            paths.append(SimPath(
                regime_labels=regime_seq,
                risk_multipliers=mult_seq,
                price_path=price_seq,
                cumulative_return=cum_ret,
            ))

        return paths

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(
        self,
        paths:         list[SimPath],
        snap:          RegimeSnapshot,
        symbol:        str,
        horizon:       int,
        n_paths:       int,
        current_price: float,
    ) -> SimulationResult:
        all_regimes = self._all_regimes
        n_regimes   = len(all_regimes)

        # Per-step regime occupancy
        occupancy = []
        for step in range(horizon):
            counts = {}
            for p in paths:
                lbl = p.regime_labels[step]
                counts[lbl] = counts.get(lbl, 0) + 1
            occupancy.append({lbl: counts.get(lbl, 0) / n_paths for lbl in all_regimes})

        # Per-step expected risk multiplier
        exp_mult = [
            float(np.mean([p.risk_multipliers[step] for p in paths]))
            for step in range(horizon)
        ]

        # Price fan
        price_matrix = np.array([p.price_path[1:] for p in paths])  # (n_paths, horizon)
        p5   = np.percentile(price_matrix, 5,  axis=0).tolist()
        p25  = np.percentile(price_matrix, 25, axis=0).tolist()
        p50  = np.percentile(price_matrix, 50, axis=0).tolist()
        p75  = np.percentile(price_matrix, 75, axis=0).tolist()
        p95  = np.percentile(price_matrix, 95, axis=0).tolist()

        # Terminal return distribution
        terminal_returns = np.array([p.cumulative_return for p in paths])
        ret_mean  = float(np.mean(terminal_returns))
        ret_std   = float(np.std(terminal_returns, ddof=1))
        var95     = float(-np.percentile(terminal_returns, 5))
        tail      = terminal_returns[terminal_returns <= -var95]
        cvar95    = float(-tail.mean()) if len(tail) > 0 else var95

        # Terminal regime mode
        terminal_counts: dict[str, int] = {}
        for p in paths:
            lbl = p.regime_labels[-1]
            terminal_counts[lbl] = terminal_counts.get(lbl, 0) + 1
        terminal_mode = max(terminal_counts, key=terminal_counts.get)

        # Path-level stats
        pct_pos = float(np.mean(terminal_returns > 0))
        dd_means = []
        for p in paths:
            arr  = np.array(p.price_path)
            peak = np.maximum.accumulate(arr)
            dd   = float(np.min((arr - peak) / peak))
            dd_means.append(dd)
        mdd_mean = float(np.mean(dd_means))

        return SimulationResult(
            symbol=symbol,
            horizon=horizon,
            n_paths=n_paths,
            current_regime=snap.regime_label,
            current_price=current_price,
            regime_occupancy=occupancy,
            expected_risk_mult=exp_mult,
            price_median=p50,
            price_p5=p5,
            price_p25=p25,
            price_p75=p75,
            price_p95=p95,
            return_mean=round(ret_mean, 6),
            return_std=round(ret_std, 6),
            return_var95=round(var95, 6),
            return_cvar95=round(cvar95, 6),
            terminal_regime_mode=terminal_mode,
            pct_paths_positive=round(pct_pos, 4),
            max_drawdown_mean=round(mdd_mean, 6),
        )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_snap(regime_label: str) -> RegimeSnapshot:
    """Create a minimal RegimeSnapshot for risk_multiplier lookup."""
    parts     = regime_label.split("_vol_")
    vol_state = parts[0] if len(parts) == 2 else "medium"
    trend_state = parts[1] if len(parts) == 2 else "neutral"
    # Clamp to valid labels
    if vol_state   not in RegimeHMM.VOL_LABELS:   vol_state   = "med_high"
    if trend_state not in RegimeHMM.TREND_LABELS: trend_state = "bear"
    return RegimeSnapshot(
        vol_probs=[0.0]*4, trend_probs=[0.0]*4,
        vol_state=vol_state, trend_state=trend_state,
        regime_label=regime_label, confidence=1.0, n_bars_fitted=0,
    )
