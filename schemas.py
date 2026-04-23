"""Pydantic I/O models for the API."""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


# ─── Shared ───────────────────────────────────────────────────────────────────

class PricePayload(BaseModel):
    prices: list[float] = Field(..., min_length=51, description="Ordered price series (OHLCV close preferred). Min 51 bars.")
    symbol: str = Field(default="UNKNOWN", description="Ticker symbol for labelling.")

    @field_validator("prices")
    @classmethod
    def no_non_positive(cls, v):
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive.")
        return v


# ─── Regime ───────────────────────────────────────────────────────────────────

class RegimeRequest(PricePayload):
    pass


class RegimeResponse(BaseModel):
    symbol: str
    vol_probs: dict[str, float]     # {"low": x, "medium": y, "high": z}
    trend_probs: dict[str, float]   # {"bear": x, "neutral": y, "bull": z}
    vol_state: str
    trend_state: str
    regime_label: str
    confidence: float
    risk_multiplier: float
    n_bars_fitted: int


# ─── Position Sizing ──────────────────────────────────────────────────────────

class SizeRequest(PricePayload):
    returns: Optional[list[float]] = Field(
        default=None,
        description="Use explicit returns instead of computing from prices.",
    )
    kelly_fraction: float = Field(default=0.5, ge=0.1, le=1.0)
    target_vol: float    = Field(default=0.15, gt=0, le=1.0, description="Annualised target vol for scaling.")
    use_regime: bool     = Field(default=True)


class SizeResponse(BaseModel):
    symbol: str
    full_kelly_f: float
    fractional_f: float
    vol_scaled_f: float
    regime_gated_f: float
    recommended_f: float
    kelly_edge: float
    sharpe_ratio: float
    regime_label: str
    regime_multiplier: float
    fraction_type: str
    notes: list[str]


# ─── Risk Analysis ────────────────────────────────────────────────────────────

class RiskRequest(SizeRequest):
    base_risk_limit: float = Field(default=0.02, gt=0, le=0.5, description="Max daily loss fraction at 1× leverage.")


class RiskResponse(BaseModel):
    symbol: str
    # Position
    recommended_f: float
    # VaR / CVaR
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    # Portfolio stats
    max_drawdown: float
    annual_vol: float
    annual_return: float
    calmar_ratio: float
    sortino_ratio: float
    # Regime
    regime_label: str
    regime_max_loss: float
    # Trade guidance
    suggested_stop: float
    suggested_tp: float
    risk_budget_used: float
    # Kelly notes
    notes: list[str]


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class FullAnalysisRequest(PricePayload):
    kelly_fraction: float = Field(default=0.5, ge=0.1, le=1.0)
    target_vol: float     = Field(default=0.15, gt=0, le=1.0)
    base_risk_limit: float = Field(default=0.02, gt=0, le=0.5)


class FullAnalysisResponse(BaseModel):
    symbol: str
    regime: RegimeResponse
    sizing: SizeResponse
    risk: RiskResponse
