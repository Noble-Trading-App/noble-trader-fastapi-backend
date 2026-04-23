"""Position sizing endpoints."""

from fastapi import APIRouter, HTTPException
from ..models.schemas import SizeRequest, SizeResponse
from ..core.regime_engine import RegimeHMM
from ..core.position_sizer import PositionSizer, PositionRequest
import numpy as np

router = APIRouter(prefix="/size", tags=["Position Sizing"])
sizer  = PositionSizer()


@router.post("/kelly", response_model=SizeResponse, summary="Regime-aware Kelly position sizing")
async def kelly_size(req: SizeRequest):
    """
    Computes fractional Kelly position size, optionally scaled by:
    - Current realised volatility vs target volatility
    - Dynamic regime risk multiplier from HMM
    """
    try:
        prices  = np.array(req.prices, dtype=float)
        returns = np.diff(np.log(prices)) if req.returns is None else np.array(req.returns, dtype=float)

        regime = None
        if req.use_regime:
            model = RegimeHMM()
            model.fit(prices)
            regime = model.predict(prices)

        result = sizer.size(PositionRequest(
            returns=returns.tolist(),
            kelly_fraction=req.kelly_fraction,
            target_vol=req.target_vol,
            regime=regime,
        ))

        return SizeResponse(
            symbol=req.symbol,
            full_kelly_f=result.full_kelly_f,
            fractional_f=result.fractional_f,
            vol_scaled_f=result.vol_scaled_f,
            regime_gated_f=result.regime_gated_f,
            recommended_f=result.recommended_f,
            kelly_edge=result.kelly_edge,
            sharpe_ratio=result.sharpe_ratio,
            regime_label=result.regime_label,
            regime_multiplier=result.regime_multiplier,
            fraction_type=result.fraction_type,
            notes=result.notes,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
