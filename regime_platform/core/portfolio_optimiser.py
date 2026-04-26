"""
Drawdown-Controlled Portfolio Optimiser
════════════════════════════════════════

Solves a regime-aware maximum Sharpe portfolio subject to a maximum
drawdown constraint. Combines three layers:

  1. Regime-adjusted expected returns
     Each asset's expected return is scaled by its current regime risk
     multiplier. Bull regimes get full weight; bear/high-vol regimes
     are penalised in the objective.

  2. Drawdown-controlled weight bounds
     Asset weight upper bounds are tightened in high-risk regimes:
       low_vol_strong_bull → max weight = min(kelly_f × 1.5, 0.40)
       high_vol_strong_bear → max weight = min(kelly_f × 0.25, 0.10)

  3. Correlation-regime overlay
     The correlation regime multiplier scales the entire portfolio's
     gross exposure (sum of weights), contracting in crisis regimes.

Optimisation problem
────────────────────
  max   w' μ_adj / √(w' Σ w)          [regime-adjusted Sharpe]
  s.t.  sum(w) = 1
        w_i ≥ 0
        w_i ≤ bound_i(regime)
        Expected_Drawdown(w) ≤ max_dd_limit

The drawdown constraint is implemented as a soft penalty term
(Cornish-Fisher CVaR proxy) since analytic drawdown constraints
are computationally expensive for general return distributions.

References
──────────
Markowitz (1952) Portfolio Selection
Grossman & Zhou (1993) Optimal Investment Strategies for Controlling Drawdowns
Magdon-Ismail & Atiya (2004) Maximum Drawdown
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import minimize
from .regime_engine import RegimeSnapshot


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class OptimisationResult:
    """Output of the portfolio optimiser."""
    symbols:              list[str]
    weights:              list[float]           # optimised portfolio weights
    regime_adj_weights:   list[float]           # after regime exposure scaling
    expected_return:      float                 # annualised, regime-adjusted
    expected_vol:         float                 # annualised portfolio vol
    sharpe_ratio:         float
    expected_max_drawdown: float                # estimated max drawdown (CF proxy)
    dd_constraint_met:    bool                  # was the DD limit satisfied?
    regime_exposure:      float                 # gross exposure scalar [0.35, 1.00]
    per_asset_bounds:     list[tuple[float,float]]
    converged:            bool
    solver_message:       str


@dataclass
class OptimiserConfig:
    max_dd_limit:         float = 0.20     # max acceptable expected drawdown [0,1]
    min_weight:           float = 0.0      # minimum asset weight (no short-selling)
    max_weight:           float = 0.40     # default maximum single asset weight
    risk_free_rate:       float = 0.04     # annualised, used for Sharpe
    annualise:            int   = 252      # trading days per year
    dd_penalty_lambda:    float = 2.0      # penalty weight for DD constraint violation
    max_iter:             int   = 1000
    tol:                  float = 1e-8


# ─── Regime weight bounds ──────────────────────────────────────────────────────

def _regime_weight_bound(snap: Optional[RegimeSnapshot], base_max: float) -> float:
    """
    Scale the max weight for an asset by its current regime state.
    In severe regimes, we tighten the upper bound regardless of Kelly.
    """
    if snap is None:
        return base_max
    mult = snap.risk_multiplier   # [0.10, 1.75]
    # Scale bound proportionally: at mult=1.75 → 100% of base_max
    #                              at mult=0.10 → 25% of base_max
    scale = np.interp(mult, [0.10, 1.75], [0.25, 1.00])
    return float(np.clip(base_max * scale, 0.01, base_max))


def _corr_regime_exposure(corr_regime: Optional[str]) -> float:
    """
    Gross exposure scalar driven by the correlation regime.
    In crisis, we scale down the entire portfolio's total weight.
    """
    return {
        "low_corr":  1.00,
        "mid_corr":  0.90,
        "high_corr": 0.70,
        "crisis":    0.50,
    }.get(corr_regime or "mid_corr", 0.90)


# ─── Optimiser ────────────────────────────────────────────────────────────────

class DrawdownOptimiser:
    """
    Regime-aware, drawdown-controlled portfolio optimiser.

    Usage
    ─────
    opt = DrawdownOptimiser()
    result = opt.optimise(
        returns_matrix=returns,          # (n_bars, n_assets)
        symbols=["SPY","QQQ","GLD","TLT"],
        asset_snaps=per_asset_snaps,     # list[RegimeSnapshot]
        corr_regime="mid_corr",          # from CorrelationRegimeDetector
    )
    print(result.weights)
    print(result.sharpe_ratio)
    """

    def __init__(self, config: Optional[OptimiserConfig] = None):
        self.config = config or OptimiserConfig()

    def optimise(
        self,
        returns_matrix: np.ndarray,          # (n_bars, n_assets) log-returns
        symbols:        list[str],
        asset_snaps:    Optional[list[Optional[RegimeSnapshot]]] = None,
        corr_regime:    Optional[str] = None,
        kelly_fractions: Optional[list[float]] = None,
    ) -> OptimisationResult:
        """
        Solve the regime-adjusted max-Sharpe portfolio with drawdown constraint.

        Parameters
        ──────────
        returns_matrix    (n_bars, n_assets) log-return history.
        symbols           Asset labels.
        asset_snaps       Per-asset regime snapshots (optional but recommended).
        corr_regime       Correlation regime label from CorrelationRegimeDetector.
        kelly_fractions   Per-asset Kelly fractions to inform weight bounds (optional).
        """
        cfg   = self.config
        n     = returns_matrix.shape[1]
        snaps = asset_snaps or [None] * n

        # ── Moment estimation ─────────────────────────────────────────────────
        mu_raw = np.mean(returns_matrix, axis=0) * cfg.annualise
        cov    = np.cov(returns_matrix.T) * cfg.annualise
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # ── Regime-adjusted expected returns ──────────────────────────────────
        regime_mults = np.array([
            snap.risk_multiplier if snap else 1.0
            for snap in snaps
        ])
        mu_adj = mu_raw * regime_mults

        # ── Per-asset weight bounds ───────────────────────────────────────────
        base_max  = cfg.max_weight
        ub_assets = [_regime_weight_bound(snap, base_max) for snap in snaps]
        bounds    = [(cfg.min_weight, ub) for ub in ub_assets]

        # ── Correlation regime exposure scalar ────────────────────────────────
        exposure  = _corr_regime_exposure(corr_regime)

        # ── Objective: negative regime-adjusted Sharpe + DD penalty ──────────
        def neg_sharpe_with_dd(w: np.ndarray) -> float:
            port_ret = float(w @ mu_adj)
            port_var = float(w @ cov @ w)
            port_vol = np.sqrt(max(port_var, 1e-12))
            sharpe   = (port_ret - cfg.risk_free_rate) / port_vol

            # Cornish-Fisher CVaR proxy as drawdown surrogate
            # E[MaxDD] ≈ CVaR(0.95) × √(T/252) for T-bar series (approximation)
            n_bars     = returns_matrix.shape[0]
            cov_daily  = np.cov(returns_matrix.T) if n > 1 else np.array([[float(np.var(returns_matrix))]])
            port_var_d = float(w @ cov_daily @ w)
            daily_vol  = np.sqrt(max(port_var_d, 1e-12))
            cf_cvar    = 2.33 * daily_vol * np.sqrt(n_bars / cfg.annualise)  # rough proxy
            dd_excess  = max(0.0, cf_cvar - cfg.max_dd_limit)
            dd_penalty = cfg.dd_penalty_lambda * dd_excess ** 2

            return -sharpe + dd_penalty

        # ── Constraints: weights sum to exposure scalar ───────────────────────
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - exposure}]
        w0          = np.full(n, exposure / n)

        result = minimize(
            neg_sharpe_with_dd,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
        )

        w_opt = np.clip(result.x, 0.0, None)
        # Re-normalise to exposure after clip
        w_sum = w_opt.sum()
        if w_sum > 1e-8:
            w_opt = w_opt / w_sum * exposure

        # ── Regime-adjusted weights (scale by per-asset mult) ─────────────────
        w_adj = w_opt * regime_mults
        w_adj_sum = w_adj.sum()
        if w_adj_sum > 1e-8:
            w_adj = w_adj / w_adj_sum * exposure

        # ── Output metrics ────────────────────────────────────────────────────
        port_ret_opt = float(w_opt @ mu_adj)
        port_var_opt = float(w_opt @ cov @ w_opt)
        port_vol_opt = float(np.sqrt(max(port_var_opt, 1e-12)))
        sharpe_opt   = (port_ret_opt - cfg.risk_free_rate) / port_vol_opt if port_vol_opt > 0 else 0.0

        cov_daily_opt  = np.cov(returns_matrix.T) if n > 1 else np.array([[float(np.var(returns_matrix))]])
        daily_vol_opt  = np.sqrt(max(float(w_opt @ cov_daily_opt @ w_opt), 1e-12))
        n_bars = returns_matrix.shape[0]
        exp_mdd = float(2.33 * daily_vol_opt * np.sqrt(n_bars / cfg.annualise))

        return OptimisationResult(
            symbols=symbols,
            weights=[round(float(x), 6) for x in w_opt],
            regime_adj_weights=[round(float(x), 6) for x in w_adj],
            expected_return=round(port_ret_opt, 6),
            expected_vol=round(port_vol_opt, 6),
            sharpe_ratio=round(sharpe_opt, 4),
            expected_max_drawdown=round(exp_mdd, 6),
            dd_constraint_met=exp_mdd <= cfg.max_dd_limit,
            regime_exposure=round(float(exposure), 4),
            per_asset_bounds=[(round(b[0], 4), round(b[1], 4)) for b in bounds],
            converged=result.success,
            solver_message=result.message,
        )
