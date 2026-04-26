"""
Regime Detection Engine — Hidden Markov Model via hmmlearn.
Detects volatility (Low / Medium-Low / Medium-High / High) and
trend (Strong-Bear / Bear / Bull / Strong-Bull) regimes.

Upgraded from 3 → 4 states per dimension for finer-grained risk segmentation.
The 4×4 grid yields 16 distinct regime labels, allowing the position sizer
to react more precisely to transitional market conditions.
"""

from __future__ import annotations
import numpy as np
from hmmlearn import hmm
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RegimeSnapshot:
    """Point-in-time regime state for one symbol."""
    vol_probs:   list[float]   # [P(low), P(med_low), P(med_high), P(high)]
    trend_probs: list[float]   # [P(strong_bear), P(bear), P(bull), P(strong_bull)]
    vol_state:   str           # dominant volatility label
    trend_state: str           # dominant trend label
    regime_label: str          # combined label e.g. "low_vol_bull"
    confidence:  float         # max(vol) * max(trend)
    n_bars_fitted: int

    @property
    def risk_multiplier(self) -> float:
        """
        Risk scaling factor [0.10 – 1.75] across the 4×4 regime grid.

        Vol scale:
          low       → 1.75  (quietest — max leverage headroom)
          med_low   → 1.20
          med_high  → 0.70
          high      → 0.35  (most volatile — strongly defensive)

        Trend scale:
          strong_bull → 1.00
          bull        → 0.88
          bear        → 0.60
          strong_bear → 0.40  (confirmed downtrend — minimal exposure)

        Combined = clip(vol_scale × trend_scale, 0.10, 1.75)
        """
        vol_scale = {
            "low":      1.75,
            "med_low":  1.20,
            "med_high": 0.70,
            "high":     0.35,
        }[self.vol_state]

        trend_scale = {
            "strong_bull": 1.00,
            "bull":        0.88,
            "bear":        0.60,
            "strong_bear": 0.40,
        }[self.trend_state]

        return round(max(0.10, min(1.75, vol_scale * trend_scale)), 4)


# ─── HMM Wrapper ─────────────────────────────────────────────────────────────

class RegimeHMM:
    """
    Dual HMM: separate 4-state Gaussian HMMs for volatility and trend regimes.

    States per dimension
    ─────────────────────
    Volatility  : low | med_low | med_high | high
    Trend       : strong_bear | bear | bull | strong_bull

    Both models use posterior state probabilities (forward-backward algorithm),
    not just Viterbi hard assignments, so confidence scores are smooth and
    differentiable. States are auto-labelled by sorting HMM component means —
    no manual mapping required.
    """

    VOL_LABELS   = ["low", "med_low", "med_high", "high"]
    TREND_LABELS = ["strong_bear", "bear", "bull", "strong_bull"]

    def __init__(
        self,
        n_components: int = 4,
        n_iter: int = 100,
        covariance_type: str = "diag",
    ):
        self.n_components    = n_components
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self._vol_model:    Optional[hmm.GaussianHMM] = None
        self._trend_model:  Optional[hmm.GaussianHMM] = None
        self._vol_label_map:   dict[int, str] = {}
        self._trend_label_map: dict[int, str] = {}
        self.fitted = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray) -> "RegimeHMM":
        """
        Fit both HMMs on a price series (1-D numpy array).
        Requires at least 4 × n_components bars for stable EM convergence
        (minimum 80 at the default n_components=4; enforced at 80).
        """
        min_bars = max(50, 4 * self.n_components * 5)   # 80 for n=4
        if len(prices) < min_bars:
            raise ValueError(
                f"Need at least {min_bars} price bars for {self.n_components}-state HMM. "
                f"Got {len(prices)}."
            )

        returns        = np.diff(np.log(prices))
        vol_features   = self._vol_features(returns)
        trend_features = self._trend_features(returns)

        self._vol_model   = self._build_hmm()
        self._trend_model = self._build_hmm()

        self._vol_model.fit(vol_features)
        self._trend_model.fit(trend_features)

        # Sanitize degenerate parameters before predict() is called
        self._sanitize_model(self._vol_model)
        self._sanitize_model(self._trend_model)

        self._vol_label_map   = self._label_vol_states(self._vol_model)
        self._trend_label_map = self._label_trend_states(self._trend_model)
        self.fitted = True
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, prices: np.ndarray) -> RegimeSnapshot:
        """Return current regime snapshot from the most recent price window."""
        if not self.fitted:
            raise RuntimeError("Call .fit() before .predict()")

        returns        = np.diff(np.log(prices))
        vol_features   = self._vol_features(returns)
        trend_features = self._trend_features(returns)

        vol_posteriors   = self._vol_model.predict_proba(vol_features)[-1]
        trend_posteriors = self._trend_model.predict_proba(trend_features)[-1]

        vol_probs   = self._reorder(vol_posteriors,   self._vol_label_map,   self.VOL_LABELS)
        trend_probs = self._reorder(trend_posteriors, self._trend_label_map, self.TREND_LABELS)

        # NaN fallback: if posteriors are degenerate (all NaN), use uniform
        if not np.all(np.isfinite(vol_probs)):
            n = len(self.VOL_LABELS)
            vol_probs = [1.0 / n] * n
        if not np.all(np.isfinite(trend_probs)):
            n = len(self.TREND_LABELS)
            trend_probs = [1.0 / n] * n

        # Re-normalise (floating point drift)
        vs = sum(vol_probs)
        if vs > 0:
            vol_probs = [p / vs for p in vol_probs]
        ts = sum(trend_probs)
        if ts > 0:
            trend_probs = [p / ts for p in trend_probs]

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
        """
        Three volatility features:
          1. 20-bar rolling realised vol   (primary sorting signal)
          2. 5-bar rolling realised vol    (short-term spike detector)
          3. |return|                      (instantaneous absolute move)
        The richer feature set helps the 4-state model distinguish
        med_low from med_high more reliably than a 2-feature version.
        """
        window_long  = 20
        window_short = 5
        rv_long  = np.array([
            np.std(returns[max(0, i - window_long):i + 1])
            for i in range(len(returns))
        ])
        rv_short = np.array([
            np.std(returns[max(0, i - window_short):i + 1])
            for i in range(len(returns))
        ])
        return np.column_stack([rv_long, rv_short, np.abs(returns)])

    @staticmethod
    def _trend_features(returns: np.ndarray) -> np.ndarray:
        """
        Three trend features:
          1. Raw return                    (instantaneous direction)
          2. 10-bar cumulative return      (short-term momentum)
          3. 30-bar cumulative return      (medium-term trend)
        Dual momentum windows help strong_bull/strong_bear states separate
        from their weaker counterparts without manual threshold tuning.
        """
        window_short = 10
        window_long  = 30
        cum_short = np.array([
            np.sum(returns[max(0, i - window_short):i + 1])
            for i in range(len(returns))
        ])
        cum_long = np.array([
            np.sum(returns[max(0, i - window_long):i + 1])
            for i in range(len(returns))
        ])
        return np.column_stack([returns, cum_short, cum_long])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_hmm(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
        )

    @staticmethod
    def _sanitize_model(model: hmm.GaussianHMM) -> None:
        """
        Fix degenerate HMM parameters that cause predict_proba() to return NaN.

        When EM collapses some states (all observations assigned to a few states),
        the other states get zero rows in transmat_ and NaN/zero in startprob_.
        This is common with strongly-structured synthetic data or short windows.

        Fixes applied in-place:
        - NaN/zero startprob_ → uniform
        - NaN/zero transmat_ rows → uniform rows
        - NaN means_ → zero
        - NaN/zero covars_ → identity-like (1e-3 diagonal)
        """
        n = model.n_components
        # startprob_
        sp = model.startprob_
        if not np.all(np.isfinite(sp)) or sp.sum() < 1e-8:
            model.startprob_ = np.full(n, 1.0 / n)
        else:
            model.startprob_ = sp / sp.sum()

        # transmat_
        for i in range(n):
            row = model.transmat_[i]
            if not np.all(np.isfinite(row)) or row.sum() < 1e-8:
                model.transmat_[i] = np.full(n, 1.0 / n)
            else:
                model.transmat_[i] = row / row.sum()

        # means_
        if not np.all(np.isfinite(model.means_)):
            model.means_ = np.where(np.isfinite(model.means_), model.means_, 0.0)

        # covars_ — write to private _covars_ to bypass shape-validation setter
        # Shape: "diag" → (n_components, n_features), "full" → (n, nf, nf)
        if hasattr(model, "_covars_") and model._covars_ is not None:
            bad = ~np.isfinite(model._covars_) | (model._covars_ <= 0)
            if bad.any():
                model._covars_ = np.where(bad, 1e-3, model._covars_)

    @staticmethod
    def _label_vol_states(model: hmm.GaussianHMM) -> dict[int, str]:
        """
        Sort HMM component means on the first feature (20-bar realised vol).
        Ascending order → low, med_low, med_high, high.
        """
        means  = model.means_[:, 0]
        order  = np.argsort(means)
        labels = RegimeHMM.VOL_LABELS
        return {int(order[i]): labels[i] for i in range(len(labels))}

    @staticmethod
    def _label_trend_states(model: hmm.GaussianHMM) -> dict[int, str]:
        """
        Sort HMM component means on the third feature (30-bar cumulative return).
        Ascending order → strong_bear, bear, bull, strong_bull.
        """
        means  = model.means_[:, 2]   # 30-bar cum return = feature index 2
        order  = np.argsort(means)
        labels = RegimeHMM.TREND_LABELS
        return {int(order[i]): labels[i] for i in range(len(labels))}

    @staticmethod
    def _reorder(
        posteriors:    np.ndarray,
        label_map:     dict[int, str],
        target_labels: list[str],
    ) -> list[float]:
        """Re-order raw HMM posteriors into a fixed semantic order."""
        prob_map = {label_map[i]: float(posteriors[i]) for i in range(len(posteriors))}
        return [prob_map[lbl] for lbl in target_labels]
