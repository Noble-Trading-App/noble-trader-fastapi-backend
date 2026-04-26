"""GPU HMM capabilities and benchmark endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np
import time

from ..core.gpu_engine import gpu_capabilities, create_regime_hmm
from ..auth.jwt_auth import get_current_user, TokenData

router = APIRouter(prefix="/gpu", tags=["GPU Engine"])


class BenchmarkRequest(BaseModel):
    n_bars:       int   = Field(default=500,  ge=81,   le=50000)
    n_iterations: int   = Field(default=3,    ge=1,    le=10)
    force_cpu:    bool  = Field(default=False)


class BenchmarkResult(BaseModel):
    backend:          str
    device:           str
    n_bars:           int
    n_iterations:     int
    mean_fit_ms:      float
    mean_predict_ms:  float
    regime_label:     str
    pomegranate_available: bool
    cuda_available:   bool
    mps_available:    bool


@router.get(
    "/capabilities",
    summary="Report GPU availability and active HMM backend",
)
async def get_capabilities(user: TokenData = Depends(get_current_user)):
    """
    Returns which HMM backend will be used and what hardware is available.

    - `hmm_backend`: `"GpuRegimeHMM (pomegranate)"` or `"RegimeHMM (hmmlearn)"`
    - `active_device`: `"cuda"` | `"mps"` | `"cpu"`
    - `pomegranate_installed`: whether pomegranate ≥ 1.0 is installed
    """
    return gpu_capabilities()


@router.post(
    "/benchmark",
    response_model=BenchmarkResult,
    summary="Benchmark HMM fit + predict on synthetic data",
)
async def benchmark(
    req: BenchmarkRequest,
    user: TokenData = Depends(get_current_user),
):
    """
    Runs `n_iterations` fit+predict cycles on synthetic price data of length `n_bars`.
    Returns mean latency for fit and predict separately.

    Use `force_cpu=true` to benchmark the hmmlearn baseline regardless of GPU availability.
    Useful for measuring the speedup from GPU acceleration.
    """
    try:
        caps  = gpu_capabilities()
        model = create_regime_hmm(force_cpu=req.force_cpu)

        # Synthetic multi-regime prices
        rng    = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, req.n_bars))

        fit_times     = []
        predict_times = []
        regime_label  = "unknown"

        for _ in range(req.n_iterations):
            t0 = time.perf_counter()
            model.fit(prices)
            fit_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            snap = model.predict(prices)
            predict_times.append((time.perf_counter() - t0) * 1000)
            regime_label = snap.regime_label

        backend = "GpuRegimeHMM (pomegranate)" if hasattr(model, 'device') else "RegimeHMM (hmmlearn)"
        device  = getattr(model, 'device', 'cpu')

        return BenchmarkResult(
            backend=backend,
            device=device,
            n_bars=req.n_bars,
            n_iterations=req.n_iterations,
            mean_fit_ms=round(float(np.mean(fit_times)), 2),
            mean_predict_ms=round(float(np.mean(predict_times)), 2),
            regime_label=regime_label,
            pomegranate_available=caps["pomegranate_installed"],
            cuda_available=caps["cuda_available"],
            mps_available=caps["mps_available"],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
