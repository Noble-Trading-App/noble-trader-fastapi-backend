"""
Position Sizing Engine — v3.1
══════════════════════════════

Two complementary sizing paths, unified in one module:

  Path A — Portfolio-fraction Kelly (original, stateless)
  ────────────────────────────────────────────────────────
  Operates on historical returns. Produces a portfolio-fraction f ∈ [0,1].
  Use when you want a returns-based, regime-gated Kelly fraction for
  systematic allocation (no stop distance / point value required).

  Path B — Dynamic Masaniello (new, stateful)
  ────────────────────────────────────────────
  Operates on per-trade context: equity, stop distance, point value.
  Implements the formula:

      f_i = β × (0.5 + M_i) × Q_i × DD_i × V_i

  Where:
    β   = base risk fraction (SizingConfig.base_risk, default 0.50%)
    M_i = Masaniello progress factor = (W - w) / (N - i + 1)
          • W  = target wins for the batch
          • w  = wins so far
          • N  = batch size
          • i  = 1-based trade index
          Clamp: [0.0, 1.5]

    Q_i = Model quality factor = prob_factor × regime_factor × conf_factor
          • prob_factor  = clip((p_win - min_prob) / 0.10,  0.0, 1.5)
          • regime_factor = clip(regime_quality,              0.0, 1.5)
          • conf_factor  = clip(state_confidence,            0.0, 1.0)
          Clamp: product naturally bounded by the factor clamps above.

    DD_i = Drawdown factor = clip(1 - dd / max_dd, 0.25, 1.0)
           Floor at 0.25 keeps minimum skin-in-the-game during drawdowns.

    V_i = Volatility factor = clip(ATR_baseline / ATR_current, 0.5, 1.5)
          High vol → shrink size; low vol → allow mild expansion.

  Kelly overlay (optional):
    When use_kelly_overlay=True, the Masaniello fraction is capped at the
    fractional Kelly bound: f* = kelly_fraction × max(0, (b·p - q) / b)

  Hard clamp:
    risk_fraction = clip(f_i, min_risk, max_risk)  → [0.25%, 1.00%]

  Contract calculation:
    risk_amount     = equity × risk_fraction
    units           = risk_amount / (stop_distance_price × point_value)
    contracts       = floor(units)

  Integration with RegimeSnapshot:
  ─────────────────────────────────
  The Masaniello sizer accepts a RegimeSnapshot and extracts:
    regime_quality   ← snap.risk_multiplier        (already in [0.10, 1.75])
    state_confidence ← snap.confidence             (in [0, 1])
    regime_label     ← snap.regime_label

  Path A and Path B are independent. You can run both and use either result.
  The DynamicMasanielloSizer.size_from_snapshot() bridges them: it runs the
  Masaniello formula using the RegimeSnapshot fields for Q_i.

Design notes
────────────
• (0.5 + M_i) guarantees a minimum sizing multiplier of 0.5× even when
  M_i = 0 (e.g. target already met or first trade of a relaxed batch).
• Q_i = 0 when any sub-factor is 0. The regime_floor gate fires before
  Q is computed, so a 0-Q result always has a labelled block reason.
• BatchState tracks cumulative PnL, peak equity and drawdown across
  an N-trade batch. Call batch.reset(equity) to start a new batch.
• monte_carlo_batch() simulates n_simulations independent batch runs
  and returns distribution statistics for strategy validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from .regime_engine import RegimeSnapshot
except ImportError:
    from regime_platform.core.regime_engine import RegimeSnapshot


# ══════════════════════════════════════════════════════════════════════════════
# PATH A — Portfolio-fraction Kelly (original interface, unchanged)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class KellyResult:
    """Output of the portfolio-fraction Kelly sizer (Path A)."""

    full_kelly_f: float  # raw f* = μ/σ²
    fractional_f: float  # after kelly_fraction multiplier
    vol_scaled_f: float  # after vol-targeting
    regime_gated_f: float  # after regime risk_multiplier
    recommended_f: float  # final recommendation (= regime_gated_f)
    kelly_edge: float  # raw μ/σ² edge before clipping
    sharpe_ratio: float
    regime_label: str
    regime_multiplier: float
    fraction_type: str
    notes: list[str]


@dataclass
class PositionRequest:
    """Input to the portfolio-fraction Kelly sizer (Path A)."""

    returns: list[float]  # historical log-returns
    kelly_fraction: float = 0.5  # fractional Kelly multiplier
    target_vol: float = 0.15  # annualised target vol for scaling
    regime: RegimeSnapshot | None = None


def _kelly_continuous(mu: float, sigma2: float) -> float:
    """f* = μ / σ²  (continuous log-return Kelly formula)."""
    if sigma2 <= 1e-10:
        return 0.0
    return mu / sigma2


def _vol_scale(f: float, realised_annual_vol: float, target_vol: float) -> float:
    """Scale position fraction so realised vol matches target_vol."""
    if realised_annual_vol <= 1e-10:
        return f
    return f * (target_vol / realised_annual_vol)


class PositionSizer:
    """
    Stateless portfolio-fraction Kelly sizer (Path A).

    Call .size(PositionRequest) → KellyResult.
    All values in [0, 1] portfolio-fraction space.
    """

    ANNUALISE_FACTOR = 252

    def size(self, req: PositionRequest) -> KellyResult:
        r = np.array(req.returns, dtype=float)

        if len(r) < 2:
            raise ValueError("Need at least 2 return observations.")

        mu = float(np.mean(r))
        sigma2 = float(np.var(r, ddof=1))
        sigma = float(np.std(r, ddof=1))

        annual_mu = mu * self.ANNUALISE_FACTOR
        annual_sigma = sigma * np.sqrt(self.ANNUALISE_FACTOR)
        sharpe = annual_mu / annual_sigma if annual_sigma > 0 else 0.0

        full_kelly = float(np.clip(_kelly_continuous(mu, sigma2), 0.0, 1.0))
        frac_kelly = full_kelly * req.kelly_fraction
        vol_scaled = float(
            np.clip(_vol_scale(frac_kelly, annual_sigma, req.target_vol), 0.0, 1.0)
        )

        regime_mult = 1.0
        regime_label = "no_regime"
        if req.regime is not None:
            regime_mult = req.regime.risk_multiplier
            regime_label = req.regime.regime_label

        regime_gated = float(np.clip(vol_scaled * regime_mult, 0.0, 1.0))
        notes = self._build_notes(
            full_kelly,
            req.kelly_fraction,
            annual_sigma,
            req.target_vol,
            regime_mult,
            sharpe,
        )

        return KellyResult(
            full_kelly_f=round(full_kelly, 6),
            fractional_f=round(frac_kelly, 6),
            vol_scaled_f=round(vol_scaled, 6),
            regime_gated_f=round(regime_gated, 6),
            recommended_f=round(regime_gated, 6),
            kelly_edge=round(mu / sigma2 if sigma2 > 0 else 0.0, 6),
            sharpe_ratio=round(sharpe, 4),
            regime_label=regime_label,
            regime_multiplier=round(regime_mult, 4),
            fraction_type=f"{int(req.kelly_fraction * 100)}% Kelly",
            notes=notes,
        )

    @staticmethod
    def _build_notes(
        full_kelly, kelly_fraction, annual_sigma, target_vol, regime_mult, sharpe
    ) -> list[str]:
        notes = []
        if full_kelly > 0.5:
            notes.append(
                "Full Kelly > 50% — fractional Kelly strongly recommended to control drawdowns."
            )
        if sharpe < 0:
            notes.append(
                "Negative Sharpe — Kelly fraction will be zero (no edge detected)."
            )
        if annual_sigma > target_vol * 1.5:
            notes.append(
                f"Realised vol ({annual_sigma:.1%}) far exceeds target ({target_vol:.1%}); "
                "vol-scaling will reduce size significantly."
            )
        if regime_mult < 0.5:
            notes.append("High-risk regime detected — position size reduced by ≥50%.")
        if regime_mult > 1.2:
            notes.append(
                "Favourable low-vol bull regime — size scaled up above baseline."
            )
        return notes


# ══════════════════════════════════════════════════════════════════════════════
# PATH B — Dynamic Masaniello Sizer
# ══════════════════════════════════════════════════════════════════════════════

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SizingConfig:
    """
    Configuration for the Dynamic Masaniello sizer.

    All risk values are fractions of equity (0.005 = 0.5%).
    """

    # Base risk  β
    base_risk: float = 0.005  # 0.50% of equity — the β in f_i = β × ...
    min_risk: float = 0.0025  # hard floor: 0.25%
    max_risk: float = 0.010  # hard cap:   1.00%

    # Edge gates (pre-filter before formula runs)
    min_prob: float = 0.50  # minimum win probability to trade
    min_rr: float = 2.50  # minimum reward/risk ratio to trade

    # Drawdown limits
    max_drawdown: float = 0.10  # 10% max strategy drawdown for DD_i scaling
    batch_halt_dd: float = 0.05  # halt the batch at -5% intraday drawdown

    # Regime gate (pre-filter)
    regime_floor: float = 0.50  # minimum regime_quality to allow a trade

    # Kelly overlay
    use_kelly_overlay: bool = False
    kelly_fraction: float = 0.25  # fractional Kelly multiplier when overlay active

    # Batch parameters
    batch_size: int = 5  # N trades per batch
    target_wins: int = 3  # W target wins to hit per batch


# ── Trade context ─────────────────────────────────────────────────────────────


@dataclass
class TradeContext:
    """
    Per-trade inputs for the Masaniello sizer.

    regime_quality and state_confidence should come from a RegimeSnapshot
    (use .from_snapshot() to build this automatically).
    """

    equity: float  # current account equity in $
    stop_distance_price: float  # stop distance in price units (e.g. points)
    point_value: float  # $ value per 1 price unit (1.0 for stocks/FX)
    p_win: float  # estimated win probability [0, 1]
    reward_risk: float  # expected reward / risk ratio (R multiple)
    regime_quality: float  # 0.0 – 1.5  (from RegimeSnapshot.risk_multiplier)
    state_confidence: float  # 0.0 – 1.0  (from RegimeSnapshot.confidence)
    current_drawdown: float  # current batch drawdown as fraction [0, 1]
    atr_baseline: float  # historical ATR reference value
    atr_current: float  # current ATR value
    wins_so_far: int  # wins accumulated in current batch
    losses_so_far: int  # losses accumulated in current batch
    trade_index: int  # 1-based index within the current batch
    batch_size: int  # total batch size N
    target_wins: int  # target wins W for the batch
    direction: str  # "long" or "short"
    symbol: str = ""
    regime_label: str = ""

    @classmethod
    def from_snapshot(
        cls,
        snap: RegimeSnapshot,
        equity: float,
        stop_distance_price: float,
        point_value: float,
        p_win: float,
        reward_risk: float,
        atr_baseline: float,
        atr_current: float,
        wins_so_far: int = 0,
        losses_so_far: int = 0,
        trade_index: int = 1,
        batch_size: int = 5,
        target_wins: int = 3,
        current_drawdown: float = 0.0,
        direction: str = "long",
        symbol: str = "",
    ) -> "TradeContext":
        """
        Build a TradeContext directly from a RegimeSnapshot.

        Maps:
          regime_quality   ← snap.risk_multiplier   (0.10 – 1.75)
          state_confidence ← snap.confidence         (0 – 1)
          regime_label     ← snap.regime_label
        """
        return cls(
            equity=equity,
            stop_distance_price=stop_distance_price,
            point_value=point_value,
            p_win=float(np.clip(p_win, 0.0, 1.0)),
            reward_risk=float(max(0.0, reward_risk)),
            regime_quality=float(snap.risk_multiplier),  # [0.10, 1.75]
            state_confidence=float(snap.confidence),  # [0,    1.0]
            current_drawdown=float(np.clip(current_drawdown, 0.0, 1.0)),
            atr_baseline=float(atr_baseline),
            atr_current=float(atr_current),
            wins_so_far=wins_so_far,
            losses_so_far=losses_so_far,
            trade_index=trade_index,
            batch_size=batch_size,
            target_wins=target_wins,
            direction=direction,
            symbol=symbol,
            regime_label=snap.regime_label,
        )


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class PositionSizeResult:
    """
    Output of the Dynamic Masaniello sizer (Path B).

    All intermediate factors are exposed for inspection and logging.
    Use .summary() for a one-line human-readable representation.
    """

    allowed: bool
    risk_fraction: float  # f_i after all factors and clamps
    risk_amount: float  # equity × risk_fraction  ($)
    units: float  # risk_amount / (stop_price × point_value)
    contracts: int  # floor(units)
    masaniello_factor: float  # M_i
    quality_factor: float  # Q_i
    drawdown_factor: float  # DD_i
    volatility_factor: float  # V_i
    expected_edge: float  # E = p·R - (1-p)
    kelly_cap: Optional[float]  # fractional Kelly cap if overlay active
    reason: str

    def to_dict(self) -> dict:
        return {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in self.__dict__.items()
        }

    def summary(self) -> str:
        status = "✓ ALLOWED" if self.allowed else "✗ BLOCKED"
        return (
            f"[{status}]  "
            f"f={self.risk_fraction:.3%}  ${self.risk_amount:,.0f}  "
            f"units={self.units:.2f}  contracts={self.contracts}  "
            f"edge={self.expected_edge:+.4f}  "
            f"M={self.masaniello_factor:.3f}  Q={self.quality_factor:.3f}  "
            f"DD={self.drawdown_factor:.3f}  V={self.volatility_factor:.3f}"
            + (
                f"  kelly_cap={self.kelly_cap:.4f}"
                if self.kelly_cap is not None
                else ""
            )
        )


# ── Batch state ───────────────────────────────────────────────────────────────


@dataclass
class BatchState:
    """
    Tracks cumulative state across an N-trade batch.

    Call record() after each trade, reset(equity) at batch boundaries.
    """

    batch_num: int = 1
    trade_index: int = 1
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    history: List[dict] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as a fraction [0, 1]."""
        if self.peak_equity <= 0:
            return 0.0
        current_equity = self.peak_equity + self.pnl
        return max(0.0, (self.peak_equity - current_equity) / self.peak_equity)

    @property
    def trades_remaining(self) -> int:
        """Trades left in the current batch (before incrementing trade_index)."""
        return self.trade_index

    def record(
        self,
        result: PositionSizeResult,
        outcome: Optional[bool] = None,  # True = win, False = loss, None = pending
        pnl: float = 0.0,
        context: Optional[TradeContext] = None,
    ) -> None:
        """
        Record a completed trade into batch history.
        Updates wins, losses, pnl, peak_equity.
        """
        entry: dict = {
            "trade": self.trade_index,
            "allowed": result.allowed,
            "pnl": pnl,
            "outcome": outcome,
            **result.to_dict(),
        }
        if context:
            entry["direction"] = context.direction
            entry["regime"] = context.regime_label
            entry["p_win"] = context.p_win
            entry["rr"] = context.reward_risk
        self.history.append(entry)

        if outcome is True:
            self.wins += 1
            self.pnl += pnl
        elif outcome is False:
            self.losses += 1
            self.pnl += pnl

        self.trade_index += 1
        if self.peak_equity > 0:
            self.peak_equity = max(self.peak_equity, self.peak_equity + self.pnl)

    def reset(self, equity: float = 0.0) -> None:
        """Start a new batch. Increments batch_num; resets all per-batch counters."""
        self.batch_num += 1
        self.trade_index = 1
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.peak_equity = equity
        self.halted = False

    def __str__(self) -> str:
        return (
            f"Batch #{self.batch_num}  trade={self.trade_index}  "
            f"W={self.wins} L={self.losses}  "
            f"PnL=${self.pnl:+,.0f}  DD={self.drawdown:.2%}"
        )


# ── Core factor functions (pure, testable) ────────────────────────────────────


def _masaniello_factor(
    wins_so_far: int,
    trade_index: int,
    batch_size: int,
    target_wins: int,
) -> float:
    """
    M_i = (W - w) / (N - i + 1)

    Measures urgency: how many wins are still needed relative to
    how many trades remain.

    Returns 0.0 when target is already met (no urgency to size up).
    Returns up to 1.5 when many wins are needed and few trades remain.

    Clamp: [0.0, 1.5]
    """
    trades_left = max(1, batch_size - trade_index + 1)
    wins_needed = max(0, target_wins - wins_so_far)
    return float(np.clip(wins_needed / trades_left, 0.0, 1.5))


def _quality_factor(
    p_win: float,
    regime_quality: float,
    state_confidence: float,
    min_prob: float = 0.50,
) -> float:
    """
    Q_i = prob_factor × regime_factor × conf_factor

    prob_factor:
      Measures how far p_win is above the minimum threshold, normalised
      so that 0.10 above the floor maps to 1.0.
      e.g. p_win=0.60 → prob_factor=1.0; p_win=0.65+ → prob_factor=1.5
      Clamp: [0.0, 1.5]

    regime_factor:
      Uses regime_quality directly (risk_multiplier from RegimeSnapshot).
      Clamp: [0.0, 1.5]

    conf_factor:
      State confidence from HMM posteriors.
      Clamp: [0.0, 1.0]

    Q_i = 0 when any factor is 0 — acts as a quality gate.
    Maximum Q_i = 1.5 × 1.5 × 1.0 = 2.25 (rare; requires all factors maxed).
    """
    prob_edge = max(0.0, p_win - min_prob)
    prob_factor = float(np.clip(prob_edge / 0.10, 0.0, 1.5))
    reg_factor = float(np.clip(regime_quality, 0.0, 1.5))
    conf_factor = float(np.clip(state_confidence, 0.0, 1.0))
    return prob_factor * reg_factor * conf_factor


def _drawdown_factor(current_drawdown: float, max_drawdown: float) -> float:
    """
    DD_i = clip(1 - dd / max_dd, 0.25, 1.0)

    Scales down sizing linearly as drawdown approaches max_drawdown.
    Floor at 0.25 preserves minimum participation — the position never
    drops to zero due to drawdown alone.

    Examples (max_dd = 10%):
      dd = 0%  → 1.00 (full size)
      dd = 5%  → 0.50 (half size)
      dd = 10% → 0.25 (floor; not zero)
      dd = 15% → 0.25 (still at floor)
    """
    dd_ratio = current_drawdown / max(max_drawdown, 1e-9)
    return float(np.clip(1.0 - dd_ratio, 0.25, 1.0))


def _volatility_factor(atr_baseline: float, atr_current: float) -> float:
    """
    V_i = clip(ATR_baseline / ATR_current, 0.5, 1.5)

    Inverted ratio: higher current vol → smaller position.
    Baseline is the historical reference ATR (e.g. 20-day average).

    Examples:
      ATR_current = ATR_baseline     → V = 1.00 (neutral)
      ATR_current = 2 × ATR_baseline → V = 0.50 (floor; vol doubled)
      ATR_current = ½ × ATR_baseline → V = 1.50 (cap; vol halved)
    """
    if atr_current <= 0:
        return 1.0
    return float(np.clip(atr_baseline / atr_current, 0.5, 1.5))


def _fractional_kelly(p_win: float, reward_risk: float, kelly_fraction: float) -> float:
    """
    f* = kelly_fraction × max(0, (b·p - q) / b)
    where b = reward_risk, q = 1 - p_win.

    Returns 0.0 when the edge is non-positive.
    """
    b = reward_risk
    if b <= 0:
        return 0.0
    q = 1.0 - p_win
    kelly = (b * p_win - q) / b
    return float(max(0.0, kelly) * kelly_fraction)


def _expected_edge(p_win: float, reward_risk: float) -> float:
    """E = p·R - (1-p)  — expected value in R multiples."""
    return float(p_win * reward_risk - (1.0 - p_win))


# ── Main Masaniello sizer ─────────────────────────────────────────────────────


class DynamicMasanielloSizer:
    """
    Dynamic Masaniello position sizer (Path B).

    Implements: f_i = β × (0.5 + M_i) × Q_i × DD_i × V_i

    Maintains a BatchState across trades. Call size_trade() for each trade,
    batch.record() after outcome is known, batch.reset(equity) at batch end.

    Convenience entry-points:
      size_from_snapshot(snap, ...)  — build context from RegimeSnapshot
      run_batch(contexts, ...)       — size an entire batch in one call
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
        self.batch = BatchState()

    # ── Public factor accessors (for inspection / testing) ────────────────────

    def masaniello_factor(self, wins: int, idx: int, batch: int, target: int) -> float:
        return _masaniello_factor(wins, idx, batch, target)

    def quality_factor(
        self, p_win: float, regime_quality: float, state_confidence: float
    ) -> float:
        return _quality_factor(
            p_win, regime_quality, state_confidence, self.config.min_prob
        )

    def drawdown_factor(self, current_drawdown: float) -> float:
        return _drawdown_factor(current_drawdown, self.config.max_drawdown)

    def volatility_factor(self, atr_baseline: float, atr_current: float) -> float:
        return _volatility_factor(atr_baseline, atr_current)

    def fractional_kelly(self, p_win: float, reward_risk: float) -> float:
        return _fractional_kelly(p_win, reward_risk, self.config.kelly_fraction)

    def expected_edge(self, p_win: float, reward_risk: float) -> float:
        return _expected_edge(p_win, reward_risk)

    # ── Pre-flight gates ──────────────────────────────────────────────────────

    def _check_gates(self, ctx: TradeContext) -> Optional[str]:
        """
        Returns a block reason string if any pre-flight gate fails,
        else None (all gates passed).

        Gates are checked in priority order — most critical first.
        Each gate fires before any formula computation, so the reason
        is always specific rather than 'Q_i = 0'.
        """
        cfg = self.config

        if ctx.equity <= 0:
            return "Equity must be positive."
        if ctx.stop_distance_price <= 0:
            return "Stop distance must be positive."
        if ctx.point_value <= 0:
            return "Point value must be positive."
        if ctx.regime_quality < cfg.regime_floor:
            return (
                f"Regime quality {ctx.regime_quality:.3f} below floor "
                f"{cfg.regime_floor:.3f} — trade blocked."
            )
        if ctx.p_win < cfg.min_prob:
            return (
                f"Win probability {ctx.p_win:.3f} below minimum "
                f"{cfg.min_prob:.3f} — trade blocked."
            )
        if ctx.reward_risk < cfg.min_rr:
            return (
                f"Reward/risk {ctx.reward_risk:.2f} below minimum "
                f"{cfg.min_rr:.2f} — trade blocked."
            )
        if self.batch.halted:
            return "Batch halted — intraday drawdown limit reached."
        if self.batch.drawdown >= cfg.batch_halt_dd:
            self.batch.halted = True
            return (
                f"Batch halt triggered: drawdown {self.batch.drawdown:.2%} "
                f"≥ limit {cfg.batch_halt_dd:.2%}."
            )
        edge = _expected_edge(ctx.p_win, ctx.reward_risk)
        if edge <= 0:
            return f"Expected edge non-positive ({edge:+.4f}) — no trade."
        return None

    # ── Core sizing ───────────────────────────────────────────────────────────

    def size_trade(self, ctx: TradeContext) -> PositionSizeResult:
        """
        Compute position size for one trade.

        Formula: f_i = β × (0.5 + M_i) × Q_i × DD_i × V_i

        Returns PositionSizeResult with .allowed=False and a specific
        .reason if any gate blocks the trade before the formula runs.
        """
        cfg = self.config

        # ── Gate checks ───────────────────────────────────────────────────────
        block_reason = self._check_gates(ctx)
        if block_reason:
            return self._blocked(block_reason)

        edge = _expected_edge(ctx.p_win, ctx.reward_risk)

        # ── Compute factors ───────────────────────────────────────────────────
        M = _masaniello_factor(
            ctx.wins_so_far, ctx.trade_index, ctx.batch_size, ctx.target_wins
        )
        Q = _quality_factor(
            ctx.p_win, ctx.regime_quality, ctx.state_confidence, cfg.min_prob
        )
        DD = _drawdown_factor(ctx.current_drawdown, cfg.max_drawdown)
        V = _volatility_factor(ctx.atr_baseline, ctx.atr_current)

        # ── Core formula: f_i = β × (0.5 + M_i) × Q_i × DD_i × V_i ──────────
        risk_fraction = cfg.base_risk * (0.5 + M) * Q * DD * V

        # ── Optional Kelly overlay ────────────────────────────────────────────
        kelly_cap = None
        if cfg.use_kelly_overlay:
            kelly_cap = _fractional_kelly(
                ctx.p_win, ctx.reward_risk, cfg.kelly_fraction
            )
            if kelly_cap > 0:
                risk_fraction = min(risk_fraction, kelly_cap)
            # If Kelly says no edge (cap=0), fall back to min_risk floor below

        # ── Hard clamp [min_risk, max_risk] ───────────────────────────────────
        risk_fraction = float(np.clip(risk_fraction, cfg.min_risk, cfg.max_risk))

        # ── Contract calculation ───────────────────────────────────────────────
        risk_amount = ctx.equity * risk_fraction
        dollars_per_unit = ctx.stop_distance_price * ctx.point_value
        units = risk_amount / dollars_per_unit
        contracts = int(units)

        if contracts < 1:
            return PositionSizeResult(
                allowed=False,
                risk_fraction=risk_fraction,
                risk_amount=risk_amount,
                units=units,
                contracts=0,
                masaniello_factor=M,
                quality_factor=Q,
                drawdown_factor=DD,
                volatility_factor=V,
                expected_edge=edge,
                kelly_cap=kelly_cap,
                reason="Calculated size < 1 contract/unit.",
            )

        return PositionSizeResult(
            allowed=True,
            risk_fraction=risk_fraction,
            risk_amount=risk_amount,
            units=units,
            contracts=contracts,
            masaniello_factor=M,
            quality_factor=Q,
            drawdown_factor=DD,
            volatility_factor=V,
            expected_edge=edge,
            kelly_cap=kelly_cap,
            reason="Trade allowed.",
        )

    # ── Convenience: build context from RegimeSnapshot ────────────────────────

    def size_from_snapshot(
        self,
        snap: RegimeSnapshot,
        equity: float,
        stop_distance_price: float,
        point_value: float,
        p_win: float,
        reward_risk: float,
        atr_baseline: float,
        atr_current: float,
        direction: str = "long",
        symbol: str = "",
    ) -> PositionSizeResult:
        """
        Build a TradeContext from a RegimeSnapshot and size the trade.

        Maps snapshot fields:
          regime_quality   ← snap.risk_multiplier  (0.10 – 1.75)
          state_confidence ← snap.confidence        (0.0 – 1.0)
          regime_label     ← snap.regime_label

        Batch state (wins_so_far, trade_index, current_drawdown) is
        taken from self.batch automatically.
        """
        ctx = TradeContext.from_snapshot(
            snap=snap,
            equity=equity,
            stop_distance_price=stop_distance_price,
            point_value=point_value,
            p_win=p_win,
            reward_risk=reward_risk,
            atr_baseline=atr_baseline,
            atr_current=atr_current,
            wins_so_far=self.batch.wins,
            losses_so_far=self.batch.losses,
            trade_index=self.batch.trade_index,
            batch_size=self.config.batch_size,
            target_wins=self.config.target_wins,
            current_drawdown=self.batch.drawdown,
            direction=direction,
            symbol=symbol,
        )
        return self.size_trade(ctx)

    # ── Batch runner ──────────────────────────────────────────────────────────

    def run_batch(
        self,
        contexts: List[TradeContext],
        outcomes: Optional[List[bool]] = None,
        pnls: Optional[List[float]] = None,
    ) -> Tuple[List[PositionSizeResult], BatchState]:
        """
        Size an entire sequence of trades, updating batch state after each.

        outcomes[i] = True (win) | False (loss) | None (not yet known)
        pnls[i] = realised PnL for trade i (used to update peak_equity)

        Batch state (drawdown, wins, trade_index) is injected into each
        context automatically — do NOT pre-populate these fields.
        """
        results: List[PositionSizeResult] = []
        for i, ctx in enumerate(contexts):
            # Inject live batch state
            ctx.wins_so_far = self.batch.wins
            ctx.trade_index = self.batch.trade_index
            ctx.current_drawdown = self.batch.drawdown

            result = self.size_trade(ctx)
            results.append(result)

            outcome = outcomes[i] if outcomes and i < len(outcomes) else None
            pnl = pnls[i] if pnls and i < len(pnls) else 0.0
            self.batch.record(result, outcome, pnl, ctx)

        return results, self.batch

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _blocked(reason: str) -> PositionSizeResult:
        return PositionSizeResult(
            allowed=False,
            risk_fraction=0.0,
            risk_amount=0.0,
            units=0.0,
            contracts=0,
            masaniello_factor=0.0,
            quality_factor=0.0,
            drawdown_factor=0.0,
            volatility_factor=0.0,
            expected_edge=0.0,
            kelly_cap=None,
            reason=reason,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo batch simulation
# ══════════════════════════════════════════════════════════════════════════════


def monte_carlo_batch(
    sizer: DynamicMasanielloSizer,
    base_context: TradeContext,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation of n_simulations independent batch runs.

    Each simulation runs a full batch (config.batch_size trades), sampling
    win/loss according to base_context.p_win. Batch state is reset between
    simulations (sizer.batch is NOT mutated — a fresh BatchState per sim).

    Returns distribution statistics useful for strategy validation:
      p_target_hit   — fraction of sims where target_wins was reached
      p_halt         — fraction of sims halted by drawdown limit
      mean/median/std PnL, p10/p90 PnL
      mean/p95 max drawdown

    Note: base_context fields wins_so_far, trade_index, current_drawdown
    are overridden per-trade within each simulation.
    """
    rng = np.random.default_rng(seed)
    cfg = sizer.config

    batch_pnls = []
    batch_dds = []
    target_hits = 0
    halt_count = 0

    for _ in range(n_simulations):
        equity = base_context.equity
        wins = 0
        pnl_total = 0.0
        peak = equity
        halted = False

        for t in range(cfg.batch_size):
            # Build per-trade context with live sim state
            ctx = TradeContext(**{k: v for k, v in base_context.__dict__.items()})
            ctx.trade_index = t + 1
            ctx.wins_so_far = wins
            ctx.losses_so_far = t - wins
            ctx.current_drawdown = (
                max(0.0, (peak - (equity + pnl_total)) / peak) if peak > 0 else 0.0
            )

            result = sizer.size_trade(ctx)
            if not result.allowed:
                halted = True
                break

            won = bool(rng.random() < ctx.p_win)
            if won:
                trade_pnl = result.risk_amount * ctx.reward_risk
                wins += 1
            else:
                trade_pnl = -result.risk_amount

            pnl_total += trade_pnl
            peak = max(peak, equity + pnl_total)

        batch_pnls.append(pnl_total)
        dd = max(0.0, (peak - (equity + pnl_total)) / peak) if peak > 0 else 0.0
        batch_dds.append(dd)

        if wins >= cfg.target_wins:
            target_hits += 1
        if halted:
            halt_count += 1

    arr = np.array(batch_pnls)
    dds = np.array(batch_dds)

    return {
        "n_simulations": n_simulations,
        "p_target_hit": round(target_hits / n_simulations, 4),
        "p_halt": round(halt_count / n_simulations, 4),
        "mean_pnl": round(float(arr.mean()), 2),
        "median_pnl": round(float(np.median(arr)), 2),
        "std_pnl": round(float(arr.std()), 2),
        "p10_pnl": round(float(np.percentile(arr, 10)), 2),
        "p90_pnl": round(float(np.percentile(arr, 90)), 2),
        "mean_max_dd": round(float(dds.mean()), 4),
        "p95_max_dd": round(float(np.percentile(dds, 95)), 4),
        "sharpe_approx": round(float(arr.mean() / (arr.std() + 1e-10)), 4),
    }
