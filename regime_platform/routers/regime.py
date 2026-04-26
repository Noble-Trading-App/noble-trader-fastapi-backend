"""Regime detection endpoints."""

from fastapi import APIRouter, HTTPException
from ..models.schemas import RegimeRequest, RegimeResponse
from ..core.regime_engine import RegimeHMM
import numpy as np

router = APIRouter(prefix="/regime", tags=["Regime Detection"])


@router.post("/detect", response_model=RegimeResponse, summary="Detect market regime from price series")
async def detect_regime(req: RegimeRequest):
    """
    Fits dual Gaussian HMMs (volatility + trend) on the supplied price series
    and returns posterior regime probabilities plus a risk multiplier.
    """
    try:
        prices = np.array(req.prices, dtype=float)
        model  = RegimeHMM()
        model.fit(prices)
        snap   = model.predict(prices)

        return RegimeResponse(
            symbol=req.symbol,
            vol_probs=dict(zip(RegimeHMM.VOL_LABELS, snap.vol_probs)),
            trend_probs=dict(zip(RegimeHMM.TREND_LABELS, snap.trend_probs)),
            vol_state=snap.vol_state,
            trend_state=snap.trend_state,
            regime_label=snap.regime_label,
            confidence=snap.confidence,
            risk_multiplier=snap.risk_multiplier,
            n_bars_fitted=snap.n_bars_fitted,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
