"""Full pipeline: regime → sizing → risk in one call."""

from fastapi import APIRouter, HTTPException
from ..models.schemas import FullAnalysisRequest, FullAnalysisResponse, RegimeResponse, SizeResponse, RiskResponse
from ..core.regime_engine import RegimeHMM
from ..core.position_sizer import PositionSizer, PositionRequest
from ..core.risk_manager import RiskManager
import numpy as np

router  = APIRouter(prefix="/analyse", tags=["Full Pipeline"])
sizer   = PositionSizer()
riskman = RiskManager()


@router.post("/full", response_model=FullAnalysisResponse, summary="One-shot: regime + Kelly sizing + risk metrics")
async def full_analysis(req: FullAnalysisRequest):
    """
    Runs the complete pipeline in a single request:
    1. Fit HMM regime model → get vol/trend posteriors
    2. Compute regime-gated Kelly position size
    3. Compute VaR, CVaR, drawdown, and trade guidance
    Returns a nested JSON with all three result objects.
    """
    try:
        prices  = np.array(req.prices, dtype=float)
        returns = np.diff(np.log(prices))

        # 1. Regime
        model  = RegimeHMM()
        model.fit(prices)
        snap   = model.predict(prices)

        regime_resp = RegimeResponse(
            symbol=req.symbol,
            vol_probs=dict(zip(["low","medium","high"], snap.vol_probs)),
            trend_probs=dict(zip(["bear","neutral","bull"], snap.trend_probs)),
            vol_state=snap.vol_state,
            trend_state=snap.trend_state,
            regime_label=snap.regime_label,
            confidence=snap.confidence,
            risk_multiplier=snap.risk_multiplier,
            n_bars_fitted=snap.n_bars_fitted,
        )

        # 2. Sizing
        size_result = sizer.size(PositionRequest(
            returns=returns.tolist(),
            kelly_fraction=req.kelly_fraction,
            target_vol=req.target_vol,
            regime=snap,
        ))
        size_resp = SizeResponse(
            symbol=req.symbol,
            full_kelly_f=size_result.full_kelly_f,
            fractional_f=size_result.fractional_f,
            vol_scaled_f=size_result.vol_scaled_f,
            regime_gated_f=size_result.regime_gated_f,
            recommended_f=size_result.recommended_f,
            kelly_edge=size_result.kelly_edge,
            sharpe_ratio=size_result.sharpe_ratio,
            regime_label=size_result.regime_label,
            regime_multiplier=size_result.regime_multiplier,
            fraction_type=size_result.fraction_type,
            notes=size_result.notes,
        )

        # 3. Risk
        risk = riskman.analyse(
            returns=returns,
            recommended_f=size_result.recommended_f,
            regime=snap,
            base_risk_limit=req.base_risk_limit,
        )
        risk_resp = RiskResponse(
            symbol=req.symbol,
            recommended_f=size_result.recommended_f,
            var_95=risk.var_95,
            var_99=risk.var_99,
            cvar_95=risk.cvar_95,
            cvar_99=risk.cvar_99,
            max_drawdown=risk.max_drawdown,
            annual_vol=risk.annual_vol,
            annual_return=risk.annual_return,
            calmar_ratio=risk.calmar_ratio,
            sortino_ratio=risk.sortino_ratio,
            regime_label=risk.regime_label,
            regime_max_loss=risk.regime_max_loss,
            suggested_stop=risk.suggested_stop,
            suggested_tp=risk.suggested_tp,
            risk_budget_used=risk.risk_budget_used,
            notes=size_result.notes,
        )

        return FullAnalysisResponse(
            symbol=req.symbol,
            regime=regime_resp,
            sizing=size_resp,
            risk=risk_resp,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
