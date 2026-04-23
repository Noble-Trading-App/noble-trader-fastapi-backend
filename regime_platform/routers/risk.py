"""Risk analysis endpoints."""

from fastapi import APIRouter, HTTPException
from ..models.schemas import RiskRequest, RiskResponse
from ..core.regime_engine import RegimeHMM
from ..core.position_sizer import PositionSizer, PositionRequest
from ..core.risk_manager import RiskManager
import numpy as np

router   = APIRouter(prefix="/risk", tags=["Risk Analysis"])
sizer    = PositionSizer()
riskman  = RiskManager()


@router.post("/analyse", response_model=RiskResponse, summary="Full risk analysis: VaR, CVaR, drawdown, stop/TP suggestions")
async def risk_analyse(req: RiskRequest):
    """
    Computes comprehensive risk metrics including parametric + historical VaR/CVaR,
    max drawdown, Sortino & Calmar ratios, and regime-adjusted stop-loss/take-profit.
    """
    try:
        prices  = np.array(req.prices, dtype=float)
        returns = np.diff(np.log(prices)) if req.returns is None else np.array(req.returns, dtype=float)

        regime = None
        if req.use_regime:
            model = RegimeHMM()
            model.fit(prices)
            regime = model.predict(prices)

        size_result = sizer.size(PositionRequest(
            returns=returns.tolist(),
            kelly_fraction=req.kelly_fraction,
            target_vol=req.target_vol,
            regime=regime,
        ))

        risk = riskman.analyse(
            returns=returns,
            recommended_f=size_result.recommended_f,
            regime=regime,
            base_risk_limit=req.base_risk_limit,
        )

        return RiskResponse(
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
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
