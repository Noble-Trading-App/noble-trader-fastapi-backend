"""
Position Sizing Engine
──────────────────────
Implements:
  1. Full Kelly Criterion  (classic closed-form for continuous returns)
  2. Fractional Kelly       (half/quarter Kelly for drawdown control)
  3. Volatility-Scaled Kelly (adjusts f* by current realised vol)
  4. Regime-Gated Sizing    (scales final position by regime risk_multiplier)

All calculations are done in portfolio-fraction space (0 → 1).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal
from .regime_engine import RegimeSnapshot


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    full_kelly_f:     float     # raw Kelly fraction
    fractional_f:     float     # after Kelly fraction applied
    vol_scaled_f:     float     # after volatility normalisation
    regime_gated_f:   float     # after regime risk multiplier
    recommended_f:    float     # final recommendation (regime_gated)
    kelly_edge:       float     # mu / sigma^2
    sharpe_ratio:     float
    regime_label:     str
    regime_multiplier: float
    fraction_type:    str
    notes:            list[str]


@dataclass
class PositionRequest:
    returns: list[float]          # historical returns (e.g. daily)
    kelly_fraction: float = 0.5   # fractional Kelly multiplier [0.25, 0.5, 1.0]
    target_vol: float = 0.15      # annualised target volatility for vol scaling
    regime: RegimeSnapshot | None = None


# ─── Core Math ────────────────────────────────────────────────────────────────

def _kelly_continuous(mu: float, sigma2: float) -> float:
    """f* = mu / sigma^2  (continuous log-return Kelly formula)."""
    if sigma2 <= 1e-10:
        return 0.0
    return mu / sigma2


def _vol_scale(f: float, realised_annual_vol: float, target_vol: float) -> float:
    """Scale position so realised vol matches target_vol."""
    if realised_annual_vol <= 1e-10:
        return f
    return f * (target_vol / realised_annual_vol)


# ─── Main Sizer ───────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Stateless position sizer. Call .size(request) to get KellyResult.
    Optionally inject a RegimeSnapshot to apply regime-aware gating.
    """

    ANNUALISE_FACTOR = 252  # trading days

    def size(self, req: PositionRequest) -> KellyResult:
        r = np.array(req.returns, dtype=float)

        if len(r) < 2:
            raise ValueError("Need at least 2 return observations.")

        mu    = float(np.mean(r))
        sigma2 = float(np.var(r, ddof=1))
        sigma  = float(np.std(r, ddof=1))

        # Annualised metrics
        annual_mu    = mu * self.ANNUALISE_FACTOR
        annual_sigma = sigma * np.sqrt(self.ANNUALISE_FACTOR)
        sharpe       = annual_mu / annual_sigma if annual_sigma > 0 else 0.0

        # Kelly fractions
        full_kelly = np.clip(_kelly_continuous(mu, sigma2), 0.0, 1.0)
        frac_kelly = full_kelly * req.kelly_fraction
        vol_scaled = np.clip(_vol_scale(frac_kelly, annual_sigma, req.target_vol), 0.0, 1.0)

        # Regime gating
        regime_mult  = 1.0
        regime_label = "no_regime"
        if req.regime is not None:
            regime_mult  = req.regime.risk_multiplier
            regime_label = req.regime.regime_label

        regime_gated = np.clip(vol_scaled * regime_mult, 0.0, 1.0)

        notes = self._build_notes(full_kelly, req.kelly_fraction, annual_sigma, req.target_vol, regime_mult, sharpe)

        return KellyResult(
            full_kelly_f=round(float(full_kelly), 6),
            fractional_f=round(float(frac_kelly), 6),
            vol_scaled_f=round(float(vol_scaled), 6),
            regime_gated_f=round(float(regime_gated), 6),
            recommended_f=round(float(regime_gated), 6),
            kelly_edge=round(float(mu / sigma2) if sigma2 > 0 else 0.0, 6),
            sharpe_ratio=round(float(sharpe), 4),
            regime_label=regime_label,
            regime_multiplier=round(float(regime_mult), 4),
            fraction_type=f"{int(req.kelly_fraction * 100)}% Kelly",
            notes=notes,
        )

    @staticmethod
    def _build_notes(full_kelly, kelly_fraction, annual_sigma, target_vol, regime_mult, sharpe) -> list[str]:
        notes = []
        if full_kelly > 0.5:
            notes.append("Full Kelly > 50% — fractional Kelly strongly recommended to control drawdowns.")
        if sharpe < 0:
            notes.append("Negative Sharpe — Kelly fraction will be zero (no edge detected).")
        if annual_sigma > target_vol * 1.5:
            notes.append(f"Realised vol ({annual_sigma:.1%}) far exceeds target ({target_vol:.1%}); vol-scaling will reduce size significantly.")
        if regime_mult < 0.5:
            notes.append("High-risk regime detected — position size reduced by ≥50%.")
        if regime_mult > 1.2:
            notes.append("Favourable low-vol bull regime — size scaled up above baseline.")
        return notes
