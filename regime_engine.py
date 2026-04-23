"""
Regime Detection Engine — Hidden Markov Model via hmmlearn.
Detects volatility (Low / Medium / High) and trend (Bear / Neutral / Bull) regimes.
"""

from __future__ import annotations
import numpy as np
from hmmlearn import hmm
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RegimeSnapshot:
    """Point-in-time regime state for one symbol."""
    vol_probs: list[float]          # [P(low), P(med), P(high)]
    trend_probs: list[float]        # [P(bear), P(neutral), P(bull)]
    vol_state: str                  # dominant volatility label
    trend_state: str                # dominant trend label
    regime_label: str               # combined label e.g. "low_vol_bull"
    confidence: float               # max(vol) * max(trend)
    n_bars_fitted: int

    @property
    def risk_multiplier(self) -> float:
        """
        Returns a risk scaling factor [0.25 – 1.5] based on current regime.
        High-vol bear  → 0.25 (very defensive)
        Low-vol bull   → 1.50 (full risk-on)
        """
        vol_scale = {"low": 1.5, "medium": 1.0, "high": 0.5}[self.vol_state]
        trend_scale = {"bull": 1.0, "neutral": 0.85, "bear": 0.5}[self.trend_state]
        return round(max(0.25, min(1.5, vol_scale * trend_scale)), 4)


# ─── HMM Wrapper ─────────────────────────────────────────────────────────────

class RegimeHMM:
    """
    Dual HMM: separate 3-state Gaussian HMMs for volatility and trend regimes.
    Fits on rolling log-returns. Uses posterior state probabilities (not just
    Viterbi decode) so confidence scores are smooth and differentiable.
    """

    VOL_LABELS  = ["low", "medium", "high"]
    TREND_LABELS = ["bear", "neutral", "bull"]

    def __init__(self, n_components: int = 3, n_iter: int = 100, covariance_type: str = "full"):
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self._vol_model:   Optional[hmm.GaussianHMM] = None
        self._trend_model: Optional[hmm.GaussianHMM] = None
        self._vol_label_map:   dict[int, str] = {}
        self._trend_label_map: dict[int, str] = {}
        self.fitted = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray) -> "RegimeHMM":
        """Fit both HMMs on a price series (1-D numpy array)."""
        if len(prices) < 50:
            raise ValueError("Need at least 50 price bars to fit regime models.")

        returns = np.diff(np.log(prices))
        vol_features  = self._vol_features(returns)
        trend_features = self._trend_features(returns)

        self._vol_model   = self._build_hmm()
        self._trend_model = self._build_hmm()

        self._vol_model.fit(vol_features)
        self._trend_model.fit(trend_features)

        self._vol_label_map   = self._label_vol_states(self._vol_model)
        self._trend_label_map = self._label_trend_states(self._trend_model)
        self.fitted = True
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, prices: np.ndarray) -> RegimeSnapshot:
        """Return current regime snapshot from most recent price window."""
        if not self.fitted:
            raise RuntimeError("Call .fit() before .predict()")

        returns = np.diff(np.log(prices))
        vol_features   = self._vol_features(returns)
        trend_features = self._trend_features(returns)

        vol_posteriors   = self._vol_model.predict_proba(vol_features)[-1]
        trend_posteriors = self._trend_model.predict_proba(trend_features)[-1]

        # Re-order posteriors to match semantic labels
        vol_probs   = self._reorder(vol_posteriors,   self._vol_label_map,   self.VOL_LABELS)
        trend_probs = self._reorder(trend_posteriors, self._trend_label_map, self.TREND_LABELS)

        vol_state   = self.VOL_LABELS[int(np.argmax(vol_probs))]
        trend_state = self.TREND_LABELS[int(np.argmax(trend_probs))]

        return RegimeSnapshot(
            vol_probs=vol_probs,
            trend_probs=trend_probs,
            vol_state=vol_state,
            trend_state=trend_state,
            regime_label=f"{vol_state}_vol_{trend_state}",
            confidence=round(float(np.max(vol_probs) * np.max(trend_probs)), 4),
            n_bars_fitted=len(prices),
        )

    # ── Feature Engineering ──────────────────────────────────────────────────

    @staticmethod
    def _vol_features(returns: np.ndarray) -> np.ndarray:
        """Rolling 20-bar realised vol + absolute return."""
        window = 20
        rv = np.array([
            np.std(returns[max(0, i - window):i + 1])
            for i in range(len(returns))
        ])
        return np.column_stack([rv, np.abs(returns)])

    @staticmethod
    def _trend_features(returns: np.ndarray) -> np.ndarray:
        """Raw return + 20-bar cumulative return as trend signal."""
        window = 20
        cum = np.array([
            np.sum(returns[max(0, i - window):i + 1])
            for i in range(len(returns))
        ])
        return np.column_stack([returns, cum])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_hmm(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
        )

    @staticmethod
    def _label_vol_states(model: hmm.GaussianHMM) -> dict[int, str]:
        """Map HMM state indices to volatility labels by mean vol magnitude."""
        means = model.means_[:, 0]  # first feature = realised vol
        order = np.argsort(means)   # ascending → low, med, high
        labels = ["low", "medium", "high"]
        return {int(order[i]): labels[i] for i in range(3)}

    @staticmethod
    def _label_trend_states(model: hmm.GaussianHMM) -> dict[int, str]:
        """Map HMM state indices to trend labels by mean return."""
        means = model.means_[:, 1]  # second feature = cumulative return
        order = np.argsort(means)   # ascending → bear, neutral, bull
        labels = ["bear", "neutral", "bull"]
        return {int(order[i]): labels[i] for i in range(3)}

    @staticmethod
    def _reorder(posteriors: np.ndarray, label_map: dict[int, str], target_labels: list[str]) -> list[float]:
        """Re-order raw HMM posteriors into a fixed semantic order."""
        prob_map = {label_map[i]: float(posteriors[i]) for i in range(len(posteriors))}
        return [prob_map[lbl] for lbl in target_labels]
