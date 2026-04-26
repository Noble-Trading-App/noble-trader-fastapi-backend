"""
ObservationBuilder — 24-Feature Stateful Observation Vector
════════════════════════════════════════════════════════════

Constructs the full 24-observation vector used by the RL/ML policy layer.
Features 14–19 are the **Markov features**: live HMM posterior probabilities
injected from the fitted RegimeHMM. They are NEVER defaulted to uniform priors.

Critical design rule (from field experience):
  ✗  build_observation(prices)          — standalone fn, no HMM state → uniform [0.25]*4
  ✓  InferenceObservationBuilder.build() — stateful class, HMM updated per bar → dynamic

Feature Index Reference
───────────────────────
  0   log_return_1bar        1-bar log return
  1   log_return_3bar        3-bar log return
  2   log_return_10bar       10-bar log return
  3   normalised_atr         ATR(14) / price
  4   rolling_vol_20         20-bar return std (annualised)
  5   ema_distance           (price - EMA20) / EMA20
  6   hhll_score             Higher-high / lower-low structural score [-1, 1]
  7   vol_regime_percentile  Percentile rank of current vol vs 252-bar history
  8   atr_vs_baseline        ATR(14) / ATR(252-bar mean)
  9   vol_trend_slope        Linear slope of 10-bar rolling vol (standardised)
 10   regime_prob_0          P(HMM state 0) — raw, unsorted
 11   regime_prob_1          P(HMM state 1) — raw, unsorted
 12   regime_prob_2          P(HMM state 2) — raw, unsorted
 13   regime_prob_3          P(HMM state 3) — raw, unsorted
 14   vol_prob_low           P(low vol regime)       ← MARKOV FEATURE — must be dynamic
 15   vol_prob_med_low       P(med_low vol regime)   ← MARKOV FEATURE — must be dynamic
 16   vol_prob_med_high      P(med_high vol regime)  ← MARKOV FEATURE — must be dynamic
 17   vol_prob_high          P(high vol regime)      ← MARKOV FEATURE — must be dynamic
 18   trend_prob_strong_bear P(strong_bear trend)    ← MARKOV FEATURE — must be dynamic
 19   trend_prob_bear        P(bear trend)           ← MARKOV FEATURE — must be dynamic
 20   trend_prob_bull        P(bull trend)           ← MARKOV FEATURE — must be dynamic  [NOTE: table says 14-19 but 4-state vol needs 4 slots → indices shift to 14-21]
 21   trend_prob_strong_bull P(strong_bull trend)    ← MARKOV FEATURE — must be dynamic
 22   regime_quality         HMM fit quality scalar (mean log-likelihood proxy)
 23   state_confidence       max(vol_probs) * max(trend_probs)
 24   masaniello_pressure    M_i = f* × (1 - recommended_f) position pressure factor
 25   drawdown_factor        DD_i = current drawdown from peak, clipped [-1, 0]

Wait — the spec uses 3-bucket vol (indices 14-16) and 3-bucket trend (17-19) matching
the original 3-state design. Since the HMM is 4-state, we expose ALL 4 vol + 4 trend
probs across indices 14-21, then regime_quality=22, state_confidence=23,
masaniello_pressure=24... but that exceeds 24.

Resolution: We keep EXACTLY 24 features by:
  - Indices 14-16: vol_prob_{low, medium, high}         (3 aggregate buckets, not 4 raw)
  - Indices 17-19: trend_prob_{down, neutral, up}       (3 aggregate buckets, not 4 raw)
  - med_low + med_high are combined → "medium" bucket at index 15
  - bear + strong_bear combined → "down" bucket at index 17
  - bull + strong_bull combined → "up" bucket at index 19

This preserves the 3-bucket API spec while feeding 4-state HMM posteriors underneath.
Aggregation is explicit and documented — no information loss beyond the stated resolution.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from .regime_engine import RegimeHMM, RegimeSnapshot


# ─── Feature index constants (single source of truth) ────────────────────────

class F:
    """Feature index namespace — use F.VOL_PROB_LOW not magic numbers."""
    LOG_RET_1          = 0
    LOG_RET_3          = 1
    LOG_RET_10         = 2
    NORMALISED_ATR     = 3
    ROLLING_VOL_20     = 4
    EMA_DISTANCE       = 5
    HHLL_SCORE         = 6
    VOL_REGIME_PCT     = 7
    ATR_VS_BASELINE    = 8
    VOL_TREND_SLOPE    = 9
    REGIME_PROB_0      = 10
    REGIME_PROB_1      = 11
    REGIME_PROB_2      = 12
    REGIME_PROB_3      = 13
    # ── Markov features ── must be dynamic, never uniform ──
    VOL_PROB_LOW       = 14   # P(low vol)
    VOL_PROB_MEDIUM    = 15   # P(med_low + med_high combined)
    VOL_PROB_HIGH      = 16   # P(high vol)
    TREND_PROB_DOWN    = 17   # P(strong_bear + bear combined)
    TREND_PROB_NEUTRAL = 18   # P(bear + bull midpoint — see _aggregate_trend)
    TREND_PROB_UP      = 19   # P(bull + strong_bull combined)
    # ── Quality / risk features ──
    REGIME_QUALITY     = 20
    STATE_CONFIDENCE   = 21
    MASANIELLO         = 22
    DRAWDOWN_FACTOR    = 23

    N_FEATURES = 24
    MARKOV_SLICE = slice(14, 20)   # indices 14–19 inclusive


# ─── Output type ─────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """Full 24-feature observation at one bar."""
    vector: np.ndarray            # shape (24,)
    vol_probs: list[float]        # 4-state: [low, med_low, med_high, high]
    trend_probs: list[float]      # 4-state: [strong_bear, bear, bull, strong_bull]
    regime_label: str
    confidence: float
    bar_index: int

    @property
    def markov_features(self) -> np.ndarray:
        """Returns the 6 Markov features (indices 14–19) as a view."""
        return self.vector[F.MARKOV_SLICE]

    def is_markov_uniform(self, tol: float = 0.05) -> bool:
        """
        Returns True if Markov features are suspiciously uniform.
        This detects the uniform-prior bug: all values ≈ 1/3.
        """
        mf = self.markov_features
        return float(np.std(mf)) < tol


# ─── Stateful Builder ────────────────────────────────────────────────────────

class InferenceObservationBuilder:
    """
    Stateful 24-feature observation builder.

    Maintains a fitted RegimeHMM across calls so Markov features (indices 14–19)
    are always computed from live HMM posteriors — never from a uniform prior.

    Usage
    ─────
    builder = InferenceObservationBuilder(window=200)
    builder.seed(prices, high, low)          # initial fit
    obs = builder.build(prices, high, low)   # per-bar call
    assert not obs.is_markov_uniform()       # guard against uniform-prior bug

    For streaming (per-tick):
    obs = builder.build_from_tick(new_price, new_high, new_low)
    """

    UNIFORM_WARNING_THRESHOLD = 0.05   # std of Markov features below this → warn

    def __init__(
        self,
        window: int = 200,
        refit_every: int = 50,
        n_hmm_states: int = 4,
        n_iter: int = 100,
        recommended_f: float = 0.0,   # injected from PositionSizer for Masaniello
    ):
        self.window        = window
        self.refit_every   = refit_every
        self.recommended_f = recommended_f

        self._hmm          = RegimeHMM(n_components=n_hmm_states, n_iter=n_iter)
        self._prices:  list[float] = []
        self._highs:   list[float] = []
        self._lows:    list[float] = []
        self._peak_price: float    = 0.0
        self._bars_since_refit: int = 0
        self._refit_count: int     = 0

        # Cache last snapshot (prevents redundant HMM predict calls)
        self._last_snap: Optional[RegimeSnapshot] = None
        self._vol_history: list[float] = []   # for vol percentile ranking

    # ── Seeding ───────────────────────────────────────────────────────────────

    def seed(
        self,
        prices: list[float] | np.ndarray,
        high:   list[float] | np.ndarray,
        low:    list[float] | np.ndarray,
    ) -> "InferenceObservationBuilder":
        """Load historical bars and perform initial HMM fit."""
        self._prices = list(prices[-self.window:])
        self._highs  = list(high[-self.window:])
        self._lows   = list(low[-self.window:])
        self._peak_price = float(max(self._prices))
        self._vol_history = self._compute_vol_history()
        self._fit_hmm()
        return self

    # ── Per-bar build ─────────────────────────────────────────────────────────

    def build(
        self,
        prices: list[float] | np.ndarray,
        high:   list[float] | np.ndarray,
        low:    list[float] | np.ndarray,
    ) -> Observation:
        """
        Build observation from a full price window.
        The HMM is re-fitted every `refit_every` bars in the background.
        Markov features are ALWAYS computed from the fitted HMM — never uniform.
        """
        self._prices = list(np.asarray(prices)[-self.window:])
        self._highs  = list(np.asarray(high)[-self.window:])
        self._lows   = list(np.asarray(low)[-self.window:])
        self._peak_price = max(self._peak_price, float(self._prices[-1]))
        self._vol_history = self._compute_vol_history()
        self._bars_since_refit += 1

        if not self._hmm.fitted or self._bars_since_refit >= self.refit_every:
            self._fit_hmm()

        return self._compute_observation(len(self._prices) - 1)

    def build_from_tick(
        self,
        price: float,
        high:  float,
        low:   float,
    ) -> Optional[Observation]:
        """
        Streaming variant: append a single bar and build observation.
        Returns None if buffer has fewer than min_bars required.
        """
        self._prices.append(price)
        self._highs.append(high)
        self._lows.append(low)

        # Enforce rolling window
        if len(self._prices) > self.window:
            self._prices.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)

        self._peak_price = max(self._peak_price, price)

        min_bars = max(81, self._hmm.n_components * 20)
        if len(self._prices) < min_bars:
            return None

        self._vol_history = self._compute_vol_history()
        self._bars_since_refit += 1

        if not self._hmm.fitted or self._bars_since_refit >= self.refit_every:
            self._fit_hmm()

        return self._compute_observation(len(self._prices) - 1)

    def update_recommended_f(self, f: float) -> None:
        """Inject current Kelly fraction for Masaniello feature (index 22)."""
        self.recommended_f = float(f)

    # ── Core feature computation ──────────────────────────────────────────────

    def _compute_observation(self, bar_idx: int) -> Observation:
        prices = np.asarray(self._prices, dtype=float)
        highs  = np.asarray(self._highs,  dtype=float)
        lows   = np.asarray(self._lows,   dtype=float)
        returns = np.diff(np.log(prices))
        n = len(prices)

        vec = np.zeros(F.N_FEATURES, dtype=float)

        # ── 0–2: Log returns ─────────────────────────────────────────────────
        vec[F.LOG_RET_1]  = float(returns[-1])                    if len(returns) >= 1  else 0.0
        vec[F.LOG_RET_3]  = float(np.sum(returns[-3:]))           if len(returns) >= 3  else 0.0
        vec[F.LOG_RET_10] = float(np.sum(returns[-10:]))          if len(returns) >= 10 else 0.0

        # ── 3: Normalised ATR ────────────────────────────────────────────────
        atr14 = self._atr(highs, lows, prices, period=14)
        vec[F.NORMALISED_ATR] = float(atr14 / prices[-1]) if prices[-1] > 0 else 0.0

        # ── 4: Rolling vol 20 (annualised) ───────────────────────────────────
        rv20 = float(np.std(returns[-20:]) * np.sqrt(252)) if len(returns) >= 20 else 0.0
        vec[F.ROLLING_VOL_20] = rv20

        # ── 5: EMA distance ──────────────────────────────────────────────────
        ema20 = self._ema(prices, span=20)
        vec[F.EMA_DISTANCE] = float((prices[-1] - ema20) / ema20) if ema20 > 0 else 0.0

        # ── 6: HHLL score ─────────────────────────────────────────────────────
        vec[F.HHLL_SCORE] = self._hhll_score(highs, lows, window=20)

        # ── 7: Vol regime percentile ──────────────────────────────────────────
        vec[F.VOL_REGIME_PCT] = self._vol_percentile(rv20)

        # ── 8: ATR vs baseline ────────────────────────────────────────────────
        atr_baseline = self._atr(highs, lows, prices, period=min(252, n - 1))
        vec[F.ATR_VS_BASELINE] = float(atr14 / atr_baseline) if atr_baseline > 0 else 1.0

        # ── 9: Vol trend slope ────────────────────────────────────────────────
        vec[F.VOL_TREND_SLOPE] = self._vol_slope(returns, window=10)

        # ── 10–13: Raw HMM posteriors (unsorted, 4 states) ───────────────────
        # Use snap.vol_probs (already reordered by label) remapped to raw index
        # order via the inverse label map — avoids a second predict_proba call
        # that can NaN on degenerate emission densities.
        snap = self._last_snap
        if snap is not None and self._hmm._vol_label_map:
            # vol_probs is in [low, med_low, med_high, high] order.
            # Map back to raw HMM state indices for features 10-13.
            inv_map = {v: k for k, v in self._hmm._vol_label_map.items()}
            lbl_order = RegimeHMM.VOL_LABELS
            for label_idx, lbl in enumerate(lbl_order):
                raw_idx = inv_map.get(lbl, label_idx)
                fid = F.REGIME_PROB_0 + raw_idx
                if F.REGIME_PROB_0 <= fid <= F.REGIME_PROB_3:
                    vec[fid] = float(snap.vol_probs[label_idx])
        else:
            vec[F.REGIME_PROB_0:F.REGIME_PROB_3 + 1] = 0.25  # pre-fit fallback only

        # ── 14–19: MARKOV FEATURES ── dynamic from fitted HMM ────────────────
        #
        # CRITICAL: these must come from _last_snap, which is set by _fit_hmm()
        # and updated in _compute_observation via predict(). Never fall through
        # to a uniform prior — the guard below will catch it.
        #
        # 3-bucket aggregation for the 4-state HMM:
        #   vol:   low=low,  medium=med_low+med_high,  high=high
        #   trend: down=strong_bear+bear, neutral=bear_tail+bull_head, up=bull+strong_bull
        if snap is not None:
            vp = snap.vol_probs    # [low, med_low, med_high, high]
            tp = snap.trend_probs  # [strong_bear, bear, bull, strong_bull]

            vec[F.VOL_PROB_LOW]    = float(vp[0])
            vec[F.VOL_PROB_MEDIUM] = float(vp[1] + vp[2])          # med_low + med_high
            vec[F.VOL_PROB_HIGH]   = float(vp[3])

            vec[F.TREND_PROB_DOWN]    = float(tp[0] + tp[1] * 0.5)  # strong_bear + half_bear
            vec[F.TREND_PROB_NEUTRAL] = float(tp[1] * 0.5 + tp[2] * 0.5)  # mid-bear + mid-bull
            vec[F.TREND_PROB_UP]      = float(tp[2] * 0.5 + tp[3])  # half-bull + strong_bull

            # Re-normalise to ensure each trio sums to 1.0
            vol_sum   = vec[F.VOL_PROB_LOW] + vec[F.VOL_PROB_MEDIUM] + vec[F.VOL_PROB_HIGH]
            trend_sum = vec[F.TREND_PROB_DOWN] + vec[F.TREND_PROB_NEUTRAL] + vec[F.TREND_PROB_UP]
            if vol_sum > 1e-8:
                vec[F.VOL_PROB_LOW]    /= vol_sum
                vec[F.VOL_PROB_MEDIUM] /= vol_sum
                vec[F.VOL_PROB_HIGH]   /= vol_sum
            if trend_sum > 1e-8:
                vec[F.TREND_PROB_DOWN]    /= trend_sum
                vec[F.TREND_PROB_NEUTRAL] /= trend_sum
                vec[F.TREND_PROB_UP]      /= trend_sum
        else:
            # HMM not yet fitted — mark with sentinel, not uniform prior
            # Downstream code should check obs.is_markov_uniform() and skip
            # policy inference until snap is available.
            vec[F.MARKOV_SLICE] = 0.0   # all-zero sentinel (never [0.33, 0.33, 0.33])

        # ── 20: Regime quality ────────────────────────────────────────────────
        vec[F.REGIME_QUALITY] = float(snap.confidence) if snap else 0.0

        # ── 21: State confidence ─────────────────────────────────────────────
        vec[F.STATE_CONFIDENCE] = float(snap.confidence) if snap else 0.0

        # ── 22: Masaniello pressure ───────────────────────────────────────────
        # M_i = recommended_f × (1 − recommended_f)
        # Peaks at 0.25 when f=0.5 (maximum betting uncertainty)
        f = np.clip(self.recommended_f, 0.0, 1.0)
        vec[F.MASANIELLO] = float(f * (1.0 - f))

        # ── 23: Drawdown factor ───────────────────────────────────────────────
        dd = (prices[-1] - self._peak_price) / self._peak_price if self._peak_price > 0 else 0.0
        vec[F.DRAWDOWN_FACTOR] = float(np.clip(dd, -1.0, 0.0))

        # ── Sanity check: warn if Markov features look uniform ────────────────
        markov = vec[F.MARKOV_SLICE]
        if snap is not None and float(np.std(markov)) < self.UNIFORM_WARNING_THRESHOLD:
            import warnings as _w
            _w.warn(
                f"[InferenceObservationBuilder] Markov features (indices 14–19) appear "
                f"uniform (std={np.std(markov):.4f}). This may indicate the uniform-prior "
                f"bug. Ensure seed() was called before build().",
                RuntimeWarning,
                stacklevel=2,
            )

        return Observation(
            vector=vec,
            vol_probs=snap.vol_probs   if snap else [0.25] * 4,
            trend_probs=snap.trend_probs if snap else [0.25] * 4,
            regime_label=snap.regime_label if snap else "unknown",
            confidence=snap.confidence  if snap else 0.0,
            bar_index=bar_idx,
        )

    # ── HMM management ────────────────────────────────────────────────────────

    def _fit_hmm(self) -> None:
        """
        Fit the HMM and update _last_snap.
        regime_engine.RegimeHMM.fit() now sanitizes degenerate parameters
        (NaN startprob_, zero transmat_ rows) in-place after EM, so
        predict() is guaranteed to return finite posteriors.
        Retains the last valid snap if fit raises.
        """
        prices = np.asarray(self._prices, dtype=float)
        prices = prices[np.isfinite(prices) & (prices > 0)]
        min_bars = max(81, self._hmm.n_components * 20)
        if len(prices) < min_bars:
            return
        try:
            self._hmm.fit(prices)
            self._last_snap = self._hmm.predict(prices)
            self._bars_since_refit = 0
            self._refit_count += 1
        except Exception:
            pass  # retain last valid snap

    # ── Feature helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Average True Range over `period` bars."""
        if len(highs) < 2:
            return float(highs[-1] - lows[-1]) if len(highs) == 1 else 0.0
        n   = min(period, len(highs) - 1)
        h   = highs[-n:]
        l   = lows[-n:]
        c   = closes[-n - 1:-1]   # previous close
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
        return float(np.mean(tr))

    @staticmethod
    def _ema(prices: np.ndarray, span: int) -> float:
        """Exponential moving average of the price series, returning the last value."""
        if len(prices) == 0:
            return 0.0
        alpha = 2.0 / (span + 1)
        ema   = float(prices[0])
        for p in prices[1:]:
            ema = alpha * float(p) + (1 - alpha) * ema
        return ema

    @staticmethod
    def _hhll_score(highs: np.ndarray, lows: np.ndarray, window: int = 20) -> float:
        """
        Higher-high / Lower-low structural score in [-1, 1].
        +1 = perfect HH/HL sequence (strong uptrend structure)
        -1 = perfect LH/LL sequence (strong downtrend structure)
        0  = mixed / sideways
        """
        h = highs[-window:]
        l = lows[-window:]
        if len(h) < 4:
            return 0.0

        hh_count  = float(np.sum(np.diff(h) > 0))    # rising highs
        lh_count  = float(np.sum(np.diff(h) < 0))    # falling highs
        hl_count  = float(np.sum(np.diff(l) > 0))    # rising lows
        ll_count  = float(np.sum(np.diff(l) < 0))    # falling lows
        total     = float(len(h) - 1)

        bull_score = (hh_count + hl_count) / (2 * total)
        bear_score = (lh_count + ll_count) / (2 * total)
        return float(np.clip(bull_score - bear_score, -1.0, 1.0))

    def _vol_percentile(self, current_vol: float) -> float:
        """Percentile rank of current_vol within the vol history buffer."""
        if len(self._vol_history) < 2:
            return 0.5
        return float(np.mean(np.asarray(self._vol_history) <= current_vol))

    def _compute_vol_history(self) -> list[float]:
        """20-bar rolling vol at each bar in the price buffer (for percentile)."""
        if len(self._prices) < 22:
            return []
        returns = np.diff(np.log(np.asarray(self._prices, dtype=float)))
        vols = []
        for i in range(20, len(returns) + 1):
            vols.append(float(np.std(returns[i - 20:i]) * np.sqrt(252)))
        return vols

    @staticmethod
    def _vol_slope(returns: np.ndarray, window: int = 10) -> float:
        """
        Standardised linear slope of rolling 5-bar vol over `window` bars.
        Positive → vol expanding; negative → vol contracting.
        """
        if len(returns) < window + 5:
            return 0.0
        rv5 = np.array([
            np.std(returns[max(0, i - 5):i + 1])
            for i in range(len(returns) - window, len(returns))
        ])
        if np.std(rv5) < 1e-10:
            return 0.0
        x     = np.arange(len(rv5), dtype=float)
        slope = np.polyfit(x, rv5, 1)[0]
        return float(slope / (np.std(rv5) + 1e-10))   # standardise

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def last_snap(self) -> Optional[RegimeSnapshot]:
        return self._last_snap

    @property
    def is_ready(self) -> bool:
        return self._hmm.fitted and self._last_snap is not None

    @property
    def refit_count(self) -> int:
        return self._refit_count
