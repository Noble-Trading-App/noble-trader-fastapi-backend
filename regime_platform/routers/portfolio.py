"""Portfolio aggregation endpoints."""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Optional
from dataclasses import asdict

from ..services.portfolio_service import portfolio_service, SymbolSummary
from ..auth.jwt_auth import get_current_user, TokenData

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


class SymbolSummaryResponse(BaseModel):
    symbol:           str
    regime_label:     str
    vol_state:        str
    trend_state:      str
    confidence:       float
    risk_multiplier:  float
    recommended_f:    float
    var_95:           float
    cvar_95:          float
    sharpe_ratio:     float
    last_price:       Optional[float]
    n_bars:           int
    ready:            bool


class PortfolioResponse(BaseModel):
    n_symbols:              int
    symbols:                list[SymbolSummaryResponse]
    regime_counts:          dict[str, int]
    dominant_regime:        str
    regime_consensus:       float
    avg_risk_multiplier:    float
    avg_recommended_f:      float
    avg_sharpe:             float
    portfolio_var95:        float
    portfolio_cvar95:       float
    high_risk_count:        int
    concentration_flag:     bool
    regime_divergence_flag: bool
    active_alerts:          list[str]


@router.get(
    "",
    response_model=PortfolioResponse,
    summary="Aggregated regime + risk summary across all active sessions",
)
async def get_portfolio(
    symbols:         Optional[str]  = Query(default=None, description="Comma-separated symbols. Omit for all active sessions."),
    kelly_fraction:  float          = Query(default=0.5,  ge=0.1, le=1.0),
    target_vol:      float          = Query(default=0.15, gt=0,   le=1.0),
    base_risk_limit: float          = Query(default=0.02, gt=0,   le=0.5),
    user: TokenData = Depends(get_current_user),
):
    """
    Returns a portfolio-level view aggregating all (or specified) active
    streaming sessions:

    - Per-symbol: regime, confidence, risk_multiplier, Kelly f*, VaR95, Sharpe
    - Portfolio: dominant regime, regime consensus, avg risk multiplier,
      portfolio VaR (independence assumption: √ΣVaR²), risk flags

    Risk flags:
    - `high_risk_count`        — symbols with risk_multiplier < 0.5
    - `concentration_flag`     — any symbol with recommended_f > 40%
    - `regime_divergence_flag` — portfolio spans many different regime buckets
    - `active_alerts`          — symbols that recently changed regime

    Symbols must be seeded via POST /stream/seed before appearing here.
    """
    try:
        sym_list = [s.strip() for s in symbols.split(",")] if symbols else None
        result   = await portfolio_service.summarise(
            symbols=sym_list,
            kelly_fraction=kelly_fraction,
            target_vol=target_vol,
            base_risk_limit=base_risk_limit,
        )

        return PortfolioResponse(
            n_symbols=result.n_symbols,
            symbols=[
                SymbolSummaryResponse(
                    symbol=s.symbol,
                    regime_label=s.regime_label,
                    vol_state=s.vol_state,
                    trend_state=s.trend_state,
                    confidence=s.confidence,
                    risk_multiplier=s.risk_multiplier,
                    recommended_f=s.recommended_f,
                    var_95=s.var_95,
                    cvar_95=s.cvar_95,
                    sharpe_ratio=s.sharpe_ratio,
                    last_price=s.last_price,
                    n_bars=s.n_bars,
                    ready=s.ready,
                )
                for s in result.symbols
            ],
            regime_counts=result.regime_counts,
            dominant_regime=result.dominant_regime,
            regime_consensus=result.regime_consensus,
            avg_risk_multiplier=result.avg_risk_multiplier,
            avg_recommended_f=result.avg_recommended_f,
            avg_sharpe=result.avg_sharpe,
            portfolio_var95=result.portfolio_var95,
            portfolio_cvar95=result.portfolio_cvar95,
            high_risk_count=result.high_risk_count,
            concentration_flag=result.concentration_flag,
            regime_divergence_flag=result.regime_divergence_flag,
            active_alerts=result.active_alerts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/symbols",
    summary="List symbols with active sessions",
)
async def list_portfolio_symbols(user: TokenData = Depends(get_current_user)):
    """Returns all symbols currently registered in the session registry."""
    from ..services.registry import registry
    return {"symbols": await registry.list_symbols()}
