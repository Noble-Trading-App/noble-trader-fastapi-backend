"""Regime transition simulation endpoints."""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.jwt_auth import TokenData, get_authed_user
from ..core.simulator import RegimeSimulator, SimulationResult

router = APIRouter(prefix="/simulate", tags=["Simulation"])
simulator = RegimeSimulator()


class SimulateRequest(BaseModel):
    symbol: str = Field(default="UNKNOWN")
    prices: list[float] = Field(..., min_length=81)
    horizon: int = Field(
        default=20, ge=1, le=252, description="Forward steps to simulate"
    )
    n_paths: int = Field(default=500, ge=50, le=5000, description="Monte Carlo paths")
    seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )
    current_price: Optional[float] = Field(
        default=None, gt=0, description="Override starting price"
    )


class SimulateResponse(BaseModel):
    symbol: str
    horizon: int
    n_paths: int
    current_regime: str
    current_price: float
    # Price fan (each list has `horizon` values)
    price_p5: list[float]
    price_p25: list[float]
    price_median: list[float]
    price_p75: list[float]
    price_p95: list[float]
    # Expected regime risk multiplier per step
    expected_risk_mult: list[float]
    # Terminal statistics
    return_mean: float
    return_std: float
    return_var95: float
    return_cvar95: float
    terminal_regime_mode: str
    pct_paths_positive: float
    max_drawdown_mean: float
    # Per-step dominant regime (most likely at each step)
    step_dominant_regime: list[str]


@router.post(
    "/{symbol}",
    response_model=SimulateResponse,
    summary="Simulate regime transition paths forward in time",
)
async def simulate_symbol(
    symbol: str,
    req: SimulateRequest,
    user: TokenData = Depends(get_authed_user),
):
    """
    Fits a 4-state HMM on the supplied price series, then simulates
    `n_paths` forward paths of `horizon` bars using the fitted
    Markov transition matrix.

    Returns:
    - Price fan (p5/p25/median/p75/p95) for each forward bar
    - Expected risk multiplier at each step
    - Terminal return distribution (mean, std, VaR95, CVaR95)
    - Most likely terminal regime
    - % of paths ending with a positive return
    - Mean maximum drawdown across paths
    """
    try:
        prices = np.array(req.prices, dtype=float)
        result = simulator.simulate(
            prices=prices,
            symbol=symbol,
            horizon=req.horizon,
            n_paths=req.n_paths,
            seed=req.seed,
            current_price=req.current_price,
        )

        # Extract per-step dominant regime from occupancy
        step_dominant = [max(occ, key=occ.get) for occ in result.regime_occupancy]

        return SimulateResponse(
            symbol=result.symbol,
            horizon=result.horizon,
            n_paths=result.n_paths,
            current_regime=result.current_regime,
            current_price=result.current_price,
            price_p5=result.price_p5,
            price_p25=result.price_p25,
            price_median=result.price_median,
            price_p75=result.price_p75,
            price_p95=result.price_p95,
            expected_risk_mult=result.expected_risk_mult,
            return_mean=result.return_mean,
            return_std=result.return_std,
            return_var95=result.return_var95,
            return_cvar95=result.return_cvar95,
            terminal_regime_mode=result.terminal_regime_mode,
            pct_paths_positive=result.pct_paths_positive,
            max_drawdown_mean=result.max_drawdown_mean,
            step_dominant_regime=step_dominant,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
