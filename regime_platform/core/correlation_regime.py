"""
Multi-Asset Correlation Regime Detector
════════════════════════════════════════

Detects regimes at the portfolio level by modelling the dynamic conditional
correlation (DCC) structure across multiple assets simultaneously.

Why correlation regimes matter
──────────────────────────────
Single-asset HMMs classify each symbol independently. In a true crisis,
correlations spike toward 1.0 across all risk assets simultaneously —
diversification collapses right when you need it most. A correlation regime
detector catches this structural shift that per-symbol HMMs miss.

Architecture
────────────
1. Standardise returns for each asset (z-score via rolling EWMA vol)
2. Compute rolling Pearson correlation matrix (window = 60 bars)
3. Extract upper-triangle correlation features (vectorised)
4. Run a 4-state GaussianHMM on the correlation feature vector
5. Label states semantically by mean absolute correlation level:
     low_corr  → mean |ρ| < 0.2 (diversified, normal market)
     mid_corr  → mean |ρ| ∈ [0.2, 0.5] (moderate comovement)
     high_corr → mean |ρ| ∈ [0.5, 0.75] (stress, pre-crisis)
     crisis    → mean |ρ| > 0.75 (crash — full correlation)
6. Overlay per-asset signals from RegimeHMM with the correlation regime

Output: CorrelationSnapshot with per-asset breakdown + portfolio-level
        correlation regime + a correlation-adjusted risk multiplier.

References
──────────
Engle (2002) "Dynamic Conditional Correlation" — J. Business & Economic Stats.
Longin & Solnik (2001) "Extreme Correlation of International Equity Markets"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from .regime_engine import RegimeHMM, RegimeSnapshot


# ─── Output types ─────────────────────────────────────────────────────────────

@dataclass
class AssetRegime:
    """Per-asset regime snapshot within a portfolio context."""
    symbol:          str
    regime_label:    str
    vol_state:       str
    trend_state:     str
    confidence:      float
    risk_multiplier: float
    contribution:    float   # weight in correlation feature space


@dataclass
class CorrelationSnapshot:
    """
    Portfolio-level correlation regime state.

    The `corr_regime` field is the dominant correlation cluster:
      low_corr | mid_corr | high_corr | crisis

    `corr_risk_multiplier` scales DOWN the per-asset multipliers:
      low_corr  → 1.00 (full diversification benefit)
      mid_corr  → 0.85
      high_corr → 0.60
      crisis    → 0.35 (diversification has collapsed)

    `blended_risk_multiplier` combines per-asset and correlation regimes:
      blended = mean(per_asset_multipliers) × corr_risk_multiplier
    """
    symbols:                 list[str]
    n_assets:                int
    corr_regime:             str                  # low_corr | mid_corr | high_corr | crisis
    corr_confidence:         float
    corr_risk_multiplier:    float
    blended_risk_multiplier: float
    mean_abs_correlation:    float                # current mean |ρ| across all pairs
    correlation_matrix:      list[list[float]]    # n_assets × n_assets, last bar
    asset_regimes:           list[AssetRegime]
    n_bars_fitted:           int
    corr_probs:              dict[str, float]     # posterior per corr regime


@dataclass
class CorrelationRegimeConfig:
    window:      int   = 60     # rolling correlation window (bars)
    ewma_span:   int   = 20     # EWMA span for vol standardisation
    n_hmm_states: int = 4       # correlation HMM states
    n_iter:      int   = 100    # EM iterations


# ─── Correlation regime labels ─────────────────────────────────────────────────

CORR_LABELS       = ["low_corr", "mid_corr", "high_corr", "crisis"]
CORR_RISK_MULT    = {
    "low_corr":  1.00,
    "mid_corr":  0.85,
    "high_corr": 0.60,
    "crisis":    0.35,
}


# ─── Detector ─────────────────────────────────────────────────────────────────

class CorrelationRegimeDetector:
    """
    Multi-asset correlation regime detector.

    Usage
    ─────
    detector = CorrelationRegimeDetector()
    snap = detector.detect(
        returns_matrix,              # (n_bars, n_assets) numpy array
        symbols=["SPY", "QQQ", "GLD", "TLT"],
        asset_snaps=per_asset_snaps, # list[RegimeSnapshot] from per-asset HMMs
    )
    print(snap.corr_regime, snap.blended_risk_multiplier)
    """

    def __init__(self, config: Optional[CorrelationRegimeConfig] = None):
        self.config = config or CorrelationRegimeConfig()
        self._corr_model: Optional[RegimeHMM._build_hmm] = None  # type: ignore
        self._fitted = False
        self._label_map: dict[int, str] = {}

    def detect(
        self,
        returns_matrix: np.ndarray,      # shape (n_bars, n_assets)
        symbols:        list[str],
        asset_snaps:    Optional[list[RegimeSnapshot]] = None,
    ) -> CorrelationSnapshot:
        """
        Fit the correlation HMM and return the current correlation regime.

        Parameters
        ──────────
        returns_matrix   (n_bars, n_assets) array of log-returns.
                         Minimum bars: config.window + 10.
        symbols          Asset labels matching matrix columns.
        asset_snaps      Per-asset RegimeSnapshot list (optional).
                         If provided, blended_risk_multiplier is computed.
        """
        n_bars, n_assets = returns_matrix.shape
        min_bars = self.config.window + 10
        if n_bars < min_bars:
            raise ValueError(
                f"Need at least {min_bars} bars for correlation regime detection. Got {n_bars}."
            )
        if len(symbols) != n_assets:
            raise ValueError(f"symbols length {len(symbols)} ≠ matrix columns {n_assets}.")

        # 1. Standardise returns
        std_returns = self._standardise(returns_matrix)

        # 2. Rolling correlation features
        corr_features, corr_matrices = self._rolling_corr_features(std_returns)

        # 3. Fit / predict correlation HMM
        corr_probs, dominant_idx = self._fit_predict_corr_hmm(corr_features)

        dominant_label = self._label_map.get(dominant_idx, CORR_LABELS[1])
        corr_mult      = CORR_RISK_MULT.get(dominant_label, 0.85)

        # 4. Current correlation matrix
        last_corr = corr_matrices[-1].tolist()
        n = returns_matrix.shape[1]
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_abs_corr = float(np.mean(np.abs(corr_matrices[-1][upper_mask])))

        # 5. Confidence = max posterior
        max_prob = float(max(corr_probs.values()))

        # 6. Per-asset summaries
        asset_regimes = []
        per_asset_mults = []
        for i, sym in enumerate(symbols):
            snap = asset_snaps[i] if asset_snaps and i < len(asset_snaps) else None
            mult = snap.risk_multiplier if snap else 1.0
            per_asset_mults.append(mult)
            asset_regimes.append(AssetRegime(
                symbol=sym,
                regime_label=snap.regime_label if snap else "unknown",
                vol_state=snap.vol_state    if snap else "—",
                trend_state=snap.trend_state if snap else "—",
                confidence=snap.confidence  if snap else 0.0,
                risk_multiplier=mult,
                contribution=1.0 / n_assets,
            ))

        mean_asset_mult = float(np.mean(per_asset_mults)) if per_asset_mults else 1.0
        blended = round(mean_asset_mult * corr_mult, 4)

        return CorrelationSnapshot(
            symbols=symbols,
            n_assets=n_assets,
            corr_regime=dominant_label,
            corr_confidence=round(max_prob, 4),
            corr_risk_multiplier=corr_mult,
            blended_risk_multiplier=blended,
            mean_abs_correlation=round(mean_abs_corr, 4),
            correlation_matrix=[[round(v, 4) for v in row] for row in last_corr],
            asset_regimes=asset_regimes,
            n_bars_fitted=n_bars,
            corr_probs={CORR_LABELS[i]: round(corr_probs.get(CORR_LABELS[i], 0.0), 4)
                        for i in range(len(CORR_LABELS))},
        )

    # ── Internal methods ──────────────────────────────────────────────────────

    def _standardise(self, returns: np.ndarray) -> np.ndarray:
        """EWMA vol-standardise each asset's returns."""
        span  = self.config.ewma_span
        alpha = 2.0 / (span + 1)
        std_r = np.zeros_like(returns)
        for j in range(returns.shape[1]):
            r    = returns[:, j]
            ewma_var = np.zeros(len(r))
            ewma_var[0] = r[0] ** 2
            for t in range(1, len(r)):
                ewma_var[t] = alpha * r[t]**2 + (1 - alpha) * ewma_var[t-1]
            ewma_std = np.sqrt(np.maximum(ewma_var, 1e-10))
            std_r[:, j] = r / ewma_std
        return std_r

    def _rolling_corr_features(
        self,
        std_returns: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Compute rolling correlation matrix and extract upper-triangle features.
        Returns (features, corr_matrices) where features shape = (T', n_pairs).
        """
        window   = self.config.window
        n_bars, n_assets = std_returns.shape
        n_pairs  = n_assets * (n_assets - 1) // 2
        features = []
        corr_mats = []

        for t in range(window, n_bars + 1):
            window_r = std_returns[t - window:t]
            corr = np.corrcoef(window_r.T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            corr_mats.append(corr)
            # Upper triangle as feature vector
            upper = corr[np.triu_indices(n_assets, k=1)]
            features.append(upper)

        return np.array(features), corr_mats

    def _fit_predict_corr_hmm(
        self,
        features: np.ndarray,
    ) -> tuple[dict[str, float], int]:
        """
        Fit a 4-state Gaussian HMM on correlation features and return
        posteriors for the last bar plus the dominant state index.
        """
        from hmmlearn import hmm as hmmlearn_hmm

        n_states = self.config.n_hmm_states
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=self.config.n_iter,
            random_state=42,
        )
        model.fit(features)

        # Sanitise (same pattern as RegimeHMM._sanitize_model)
        n = model.n_components
        if not np.all(np.isfinite(model.startprob_)) or model.startprob_.sum() < 1e-8:
            model.startprob_ = np.full(n, 1.0 / n)
        else:
            model.startprob_ /= model.startprob_.sum()
        for i in range(n):
            row = model.transmat_[i]
            if not np.all(np.isfinite(row)) or row.sum() < 1e-8:
                model.transmat_[i] = np.full(n, 1.0 / n)
            else:
                model.transmat_[i] /= row.sum()

        posteriors = model.predict_proba(features)[-1]
        if not np.all(np.isfinite(posteriors)):
            posteriors = np.full(n, 1.0 / n)
        posteriors /= posteriors.sum()

        # Label states by mean absolute correlation (ascending → low..crisis)
        mean_abs = np.array([
            float(np.mean(np.abs(model.means_[k])))
            for k in range(n_states)
        ])
        order = np.argsort(mean_abs)
        label_map = {int(order[i]): CORR_LABELS[i] for i in range(n_states)}
        self._label_map = label_map
        self._fitted = True

        dominant_raw = int(np.argmax(posteriors))
        dominant_label = label_map.get(dominant_raw, CORR_LABELS[1])
        dominant_idx   = CORR_LABELS.index(dominant_label)

        # Build prob dict keyed by semantic label
        prob_dict: dict[str, float] = {}
        for raw_idx, label in label_map.items():
            prob_dict[label] = float(posteriors[raw_idx])

        return prob_dict, dominant_idx
