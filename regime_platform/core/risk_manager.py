"""
Risk Management Engine
──────────────────────
Provides:
  • Parametric & Historical VaR / CVaR
  • Max drawdown calculation
  • Dynamic risk limits driven by regime state
  • Stop-loss and take-profit suggestion engine
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats
from .regime_engine import RegimeSnapshot


@dataclass
class RiskMetrics:
    var_95:           float    # 1-day 95% Value at Risk (as fraction of portfolio)
    var_99:           float
    cvar_95:          float    # Expected Shortfall at 95%
    cvar_99:          float
    max_drawdown:     float
    annual_vol:       float
    annual_return:    float
    calmar_ratio:     float
    sortino_ratio:    float
    regime_max_loss:  float    # max acceptable loss given current regime
    suggested_stop:   float    # stop-loss suggestion in return space
    suggested_tp:     float    # take-profit suggestion
    risk_budget_used: float    # recommended_f as % of max allowed f
    regime_label:     str


class RiskManager:

    ANNUALISE = 252

    def analyse(
        self,
        returns: np.ndarray,
        recommended_f: float,
        regime: RegimeSnapshot | None = None,
        base_risk_limit: float = 0.02,   # 2% daily loss limit at 1x leverage
    ) -> RiskMetrics:

        r = np.array(returns, dtype=float)
        mu     = float(np.mean(r))
        sigma  = float(np.std(r, ddof=1))

        # ── VaR / CVaR ───────────────────────────────────────────────────────
        var_95 = float(-np.percentile(r, 5))
        var_99 = float(-np.percentile(r, 1))
        cvar_95 = float(-r[r <= -var_95].mean()) if len(r[r <= -var_95]) > 0 else var_95
        cvar_99 = float(-r[r <= -var_99].mean()) if len(r[r <= -var_99]) > 0 else var_99

        # ── Drawdown ─────────────────────────────────────────────────────────
        cum = np.cumprod(1 + r)
        roll_max = np.maximum.accumulate(cum)
        dd_series = (cum - roll_max) / roll_max
        max_dd = float(np.min(dd_series))

        # ── Annualised stats ─────────────────────────────────────────────────
        annual_vol    = sigma * np.sqrt(self.ANNUALISE)
        annual_return = mu * self.ANNUALISE
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        downside = float(np.std(r[r < 0], ddof=1)) * np.sqrt(self.ANNUALISE) if len(r[r < 0]) > 1 else annual_vol
        sortino  = annual_return / downside if downside > 0 else 0.0

        # ── Regime-adjusted limits ───────────────────────────────────────────
        regime_mult  = regime.risk_multiplier if regime else 1.0
        regime_label = regime.regime_label    if regime else "no_regime"

        regime_max_loss = base_risk_limit * regime_mult
        suggested_stop  = round(-cvar_95 * 2, 6)      # 2× CVaR95 as initial stop
        suggested_tp    = round(cvar_95 * 3, 6)        # 3:1 reward/risk target

        max_safe_f = min(1.0, regime_max_loss / (var_99 + 1e-10))
        risk_budget_used = recommended_f / max_safe_f if max_safe_f > 0 else 1.0

        return RiskMetrics(
            var_95=round(var_95, 6),
            var_99=round(var_99, 6),
            cvar_95=round(cvar_95, 6),
            cvar_99=round(cvar_99, 6),
            max_drawdown=round(max_dd, 6),
            annual_vol=round(annual_vol, 6),
            annual_return=round(annual_return, 6),
            calmar_ratio=round(calmar, 4),
            sortino_ratio=round(sortino, 4),
            regime_max_loss=round(regime_max_loss, 6),
            suggested_stop=round(suggested_stop, 6),
            suggested_tp=round(suggested_tp, 6),
            risk_budget_used=round(min(risk_budget_used, 9.99), 4),
            regime_label=regime_label,
        )
