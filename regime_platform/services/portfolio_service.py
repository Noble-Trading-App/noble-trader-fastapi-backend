"""
Portfolio Service
═════════════════

Aggregates regime state, risk metrics, and position sizing across
multiple symbols registered in the SessionRegistry.

Provides:
  • Portfolio-level regime distribution
  • Weighted average risk multiplier
  • Aggregate VaR / CVaR (assuming independence — conservative)
  • Per-symbol position summary
  • Portfolio concentration risk flag
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .registry import registry
from ..core.regime_engine import RegimeHMM
from ..core.position_sizer import PositionSizer, PositionRequest
from ..core.risk_manager import RiskManager


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class SymbolSummary:
    symbol:            str
    regime_label:      str
    vol_state:         str
    trend_state:       str
    confidence:        float
    risk_multiplier:   float
    recommended_f:     float
    var_95:            float
    cvar_95:           float
    sharpe_ratio:      float
    last_price:        Optional[float]
    n_bars:            int
    ready:             bool


@dataclass
class PortfolioSummary:
    n_symbols:                 int
    symbols:                   list[SymbolSummary]

    # Regime distribution across portfolio
    regime_counts:             dict[str, int]   # label → count
    dominant_regime:           str
    regime_consensus:          float            # fraction in dominant regime

    # Weighted averages (equal-weight)
    avg_risk_multiplier:       float
    avg_recommended_f:         float
    avg_sharpe:                float

    # Portfolio-level risk (assuming independence)
    portfolio_var95:           float            # √(Σ VaR²) — independent assets
    portfolio_cvar95:          float

    # Risk flags
    high_risk_count:           int              # symbols with risk_mult < 0.5
    concentration_flag:        bool             # any single symbol > 40% recommended_f
    regime_divergence_flag:    bool             # vol × trend entropy > threshold

    # Alert summary
    active_alerts:             list[str]        # symbols with regime_changed


# ─── Service ──────────────────────────────────────────────────────────────────

class PortfolioService:
    """
    Reads from the live SessionRegistry and computes portfolio-level analytics.
    Symbols without active sessions are skipped with a warning in the response.
    """

    def __init__(self):
        self._sizer   = PositionSizer()
        self._riskman = RiskManager()

    async def summarise(
        self,
        symbols:        Optional[list[str]] = None,
        kelly_fraction: float = 0.5,
        target_vol:     float = 0.15,
        base_risk_limit: float = 0.02,
    ) -> PortfolioSummary:
        """
        Aggregate portfolio analytics across all (or specified) active sessions.

        If `symbols` is None, all sessions in the registry are included.
        Sessions that are not yet ready (HMM not fitted) are excluded.
        """
        # Collect all active sessions
        all_stats = await registry.stats()
        target_symbols = symbols if symbols else list(all_stats.keys())

        symbol_summaries: list[SymbolSummary] = []
        active_alerts:    list[str] = []

        for sym in target_symbols:
            session = await registry.get(sym)
            if session is None or not session.is_ready:
                # Include as not-ready placeholder
                symbol_summaries.append(SymbolSummary(
                    symbol=sym, regime_label="pending",
                    vol_state="—", trend_state="—",
                    confidence=0.0, risk_multiplier=1.0,
                    recommended_f=0.0, var_95=0.0, cvar_95=0.0,
                    sharpe_ratio=0.0, last_price=None, n_bars=0, ready=False,
                ))
                continue

            snap    = session._model.predict(np.array(session.price_buffer))
            prices  = np.array(session.price_buffer)
            returns = np.diff(np.log(prices))

            size_result = self._sizer.size(PositionRequest(
                returns=returns.tolist(),
                kelly_fraction=kelly_fraction,
                target_vol=target_vol,
                regime=snap,
            ))
            risk = self._riskman.analyse(
                returns=returns,
                recommended_f=size_result.recommended_f,
                regime=snap,
                base_risk_limit=base_risk_limit,
            )

            if session._last_regime and session._last_regime != snap.regime_label:
                active_alerts.append(sym)

            symbol_summaries.append(SymbolSummary(
                symbol=sym,
                regime_label=snap.regime_label,
                vol_state=snap.vol_state,
                trend_state=snap.trend_state,
                confidence=snap.confidence,
                risk_multiplier=snap.risk_multiplier,
                recommended_f=size_result.recommended_f,
                var_95=risk.var_95,
                cvar_95=risk.cvar_95,
                sharpe_ratio=size_result.sharpe_ratio,
                last_price=float(prices[-1]) if len(prices) > 0 else None,
                n_bars=len(prices),
                ready=True,
            ))

        ready = [s for s in symbol_summaries if s.ready]
        n_ready = len(ready)

        if n_ready == 0:
            return self._empty_summary(symbol_summaries, active_alerts)

        # Regime distribution
        regime_counts: dict[str, int] = {}
        for s in ready:
            regime_counts[s.regime_label] = regime_counts.get(s.regime_label, 0) + 1
        dominant = max(regime_counts, key=regime_counts.get)
        consensus = regime_counts[dominant] / n_ready

        # Weighted averages
        avg_mult  = float(np.mean([s.risk_multiplier for s in ready]))
        avg_f     = float(np.mean([s.recommended_f   for s in ready]))
        avg_sh    = float(np.mean([s.sharpe_ratio     for s in ready]))

        # Portfolio VaR (independence assumption: √Σvar²)
        port_var  = float(np.sqrt(np.sum([s.var_95**2  for s in ready])))
        port_cvar = float(np.sqrt(np.sum([s.cvar_95**2 for s in ready])))

        # Risk flags
        high_risk = sum(1 for s in ready if s.risk_multiplier < 0.5)
        conc_flag = any(s.recommended_f > 0.40 for s in ready)

        # Regime divergence — entropy of vol_state distribution
        vol_states  = [s.vol_state   for s in ready]
        trend_states = [s.trend_state for s in ready]
        diverg_flag = self._entropy_flag(vol_states, trend_states, n_ready)

        return PortfolioSummary(
            n_symbols=len(symbol_summaries),
            symbols=symbol_summaries,
            regime_counts=regime_counts,
            dominant_regime=dominant,
            regime_consensus=round(consensus, 4),
            avg_risk_multiplier=round(avg_mult, 4),
            avg_recommended_f=round(avg_f, 4),
            avg_sharpe=round(avg_sh, 4),
            portfolio_var95=round(port_var, 6),
            portfolio_cvar95=round(port_cvar, 6),
            high_risk_count=high_risk,
            concentration_flag=conc_flag,
            regime_divergence_flag=diverg_flag,
            active_alerts=active_alerts,
        )

    @staticmethod
    def _entropy_flag(vol_states: list[str], trend_states: list[str], n: int) -> bool:
        """
        True if regime diversity is high — portfolio spans many regime buckets.
        Uses normalised entropy of vol_state distribution.
        Threshold: entropy > 0.8 × log(n_states) → flagged as divergent.
        """
        if n < 2:
            return False
        # Vol entropy
        unique_vols = list(set(vol_states))
        vol_probs   = np.array([vol_states.count(v) / n for v in unique_vols])
        vol_entropy = -np.sum(vol_probs * np.log(vol_probs + 1e-10))
        max_entropy = np.log(len(RegimeHMM.VOL_LABELS))
        return float(vol_entropy / max_entropy) > 0.75

    @staticmethod
    def _empty_summary(
        symbol_summaries: list[SymbolSummary],
        active_alerts: list[str],
    ) -> PortfolioSummary:
        return PortfolioSummary(
            n_symbols=len(symbol_summaries),
            symbols=symbol_summaries,
            regime_counts={},
            dominant_regime="pending",
            regime_consensus=0.0,
            avg_risk_multiplier=1.0,
            avg_recommended_f=0.0,
            avg_sharpe=0.0,
            portfolio_var95=0.0,
            portfolio_cvar95=0.0,
            high_risk_count=0,
            concentration_flag=False,
            regime_divergence_flag=False,
            active_alerts=active_alerts,
        )


# Singleton
portfolio_service = PortfolioService()
