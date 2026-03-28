"""Unified strategy evaluator.

Replaces one-off evaluation scripts with a reusable evaluator that takes
a :class:`SignalGenerator` and a set of :class:`EvalWindow` definitions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import TYPE_CHECKING

from .engine import backtest_signals
from .eval_windows import EvalWindow
from .models import BacktestResult, PositionType
from .pipeline import BacktestExecutionSession, prepare_market_context

if TYPE_CHECKING:
    from live.signal_generator import SignalGenerator

    from .data import BinanceClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_USD_RISK_FREE_RATE_ANNUAL = 0.0373
"""Annual USD risk-free proxy used for Sortino calculations.

Based on the U.S. Treasury 3-month constant maturity rate in the Federal
Reserve H.15 release dated March 27, 2026 (3.73% for March 26, 2026).
"""


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    max_hours: int = 24
    approximate: bool = True
    seed: int | None = None
    risk_free_rate_annual: float = DEFAULT_USD_RISK_FREE_RATE_ANNUAL


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WindowResult:
    window: EvalWindow
    backtest: BacktestResult
    signal_count: int
    short_count: int
    long_count: int


@dataclass(slots=True)
class CategorySummary:
    category: str
    windows: int
    total_pnl: float
    weekly_win_rate: float
    positive_weeks: int
    worst_week_pnl: float
    best_week_pnl: float
    total_trades: int
    short_trades: int
    long_trades: int
    trade_win_rate: float
    profit_factor: float
    sortino_ratio: float


@dataclass(slots=True)
class EvaluationReport:
    window_results: list[WindowResult]
    config: PortfolioConfig
    symbols: list[str]

    def by_category(self) -> dict[str, list[WindowResult]]:
        groups: dict[str, list[WindowResult]] = defaultdict(list)
        for wr in self.window_results:
            groups[wr.window.category].append(wr)
        return dict(groups)

    def category_summary(self, category: str) -> CategorySummary:
        wrs = [wr for wr in self.window_results if wr.window.category == category]
        return _build_summary(
            category,
            wrs,
            risk_free_rate_annual=self.config.risk_free_rate_annual,
        )

    def all_summaries(self) -> list[CategorySummary]:
        summaries: list[CategorySummary] = []
        for cat, wrs in self.by_category().items():
            summaries.append(
                _build_summary(
                    cat,
                    wrs,
                    risk_free_rate_annual=self.config.risk_free_rate_annual,
                )
            )
        return summaries

    def overall_summary(self) -> CategorySummary:
        return _build_summary(
            "ALL",
            self.window_results,
            risk_free_rate_annual=self.config.risk_free_rate_annual,
        )

    def format_table(self) -> str:
        rows = self.all_summaries() + [self.overall_summary()]
        header = (
            f"{'Category':<20} | {'Win':>7} | {'PNL':>9} | {'WR':>6} "
            f"| {'Worst':>8} | {'Best':>8} | {'Trades':>6} "
            f"| {'S/L':>9} | {'Trd WR':>6} | {'PF':>5} | {'Sort':>7}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for s in rows:
            lines.append(
                f"{s.category:<20} | {s.windows:>7} | "
                f"{s.total_pnl:>+8.2f}% | "
                f"{s.weekly_win_rate * 100:>5.1f}% | "
                f"{s.worst_week_pnl:>+7.2f}% | "
                f"{s.best_week_pnl:>+7.2f}% | "
                f"{s.total_trades:>6} | "
                f"{s.short_trades:>4}/{s.long_trades:<4} | "
                f"{s.trade_win_rate * 100:>5.1f}% | "
                f"{s.profit_factor:>5.2f} | "
                f"{s.sortino_ratio:>7.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class StrategyEvaluator:
    """Evaluate a :class:`SignalGenerator` across multiple time windows."""

    def __init__(
        self,
        symbols: list[str],
        config: PortfolioConfig = PortfolioConfig(),
        client: BinanceClient | None = None,
        cooldown_warmup: timedelta = timedelta(days=14),
    ) -> None:
        self._symbols = symbols
        self._config = config
        self._cooldown_warmup = cooldown_warmup
        if client is None:
            from .data import BinanceClient as _BinanceClient

            client = _BinanceClient()
        self._client: BinanceClient = client

    def evaluate(
        self,
        generator: SignalGenerator,
        windows: list[EvalWindow],
    ) -> EvaluationReport:
        """Run the full evaluation pipeline and return a report."""
        periods = _group_into_periods(
            sorted(windows, key=lambda w: w.start),
            gap_threshold=self._cooldown_warmup,
        )

        all_window_results: list[WindowResult] = []

        for period_windows in periods:
            earliest_start = min(w.start for w in period_windows)
            latest_end = max(w.end for w in period_windows)

            signal_start = earliest_start - self._cooldown_warmup
            signal_end = latest_end
            fetch_end = signal_end + timedelta(hours=self._config.max_hours)

            request = generator.market_data_request()
            ctx = prepare_market_context(
                self._symbols,
                signal_start,
                fetch_end,
                client=self._client,
                request=request,
                warmup_bars=100,
            )

            all_signals = generator.generate_backtest_signals(
                ctx, self._symbols, signal_start, signal_end,
            )

            session = BacktestExecutionSession(
                client=self._client,
                prepared_context=ctx,
            )

            for window in period_windows:
                w_sigs = [
                    s for s in all_signals
                    if window.start <= s.signal_date < window.end
                ]
                short_count = sum(
                    1 for s in w_sigs if s.position_type is PositionType.SHORT
                )
                long_count = len(w_sigs) - short_count

                result = backtest_signals(
                    w_sigs,
                    client=self._client,
                    max_hours=self._config.max_hours,
                    approximate=self._config.approximate,
                    seed=self._config.seed,
                    session=session,
                )

                all_window_results.append(
                    WindowResult(
                        window=window,
                        backtest=result,
                        signal_count=len(w_sigs),
                        short_count=short_count,
                        long_count=long_count,
                    )
                )

        return EvaluationReport(
            window_results=all_window_results,
            config=self._config,
            symbols=list(self._symbols),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_into_periods(
    sorted_windows: list[EvalWindow],
    gap_threshold: timedelta,
) -> list[list[EvalWindow]]:
    """Group sorted windows into contiguous periods.

    Windows whose gap is smaller than *gap_threshold* are merged into a
    single period to allow one data fetch and shared session.
    """
    if not sorted_windows:
        return []
    periods: list[list[EvalWindow]] = [[sorted_windows[0]]]
    for w in sorted_windows[1:]:
        last_end = max(pw.end for pw in periods[-1])
        if w.start - last_end <= gap_threshold:
            periods[-1].append(w)
        else:
            periods.append([w])
    return periods


def _build_summary(
    category: str,
    wrs: list[WindowResult],
    *,
    risk_free_rate_annual: float = DEFAULT_USD_RISK_FREE_RATE_ANNUAL,
) -> CategorySummary:
    if not wrs:
        return CategorySummary(
            category=category,
            windows=0,
            total_pnl=0.0,
            weekly_win_rate=0.0,
            positive_weeks=0,
            worst_week_pnl=0.0,
            best_week_pnl=0.0,
            total_trades=0,
            short_trades=0,
            long_trades=0,
            trade_win_rate=0.0,
            profit_factor=0.0,
            sortino_ratio=0.0,
        )

    week_pnls = [wr.backtest.total_pnl_pct for wr in wrs]
    positive = sum(1 for p in week_pnls if p > 0)
    total_trades = sum(wr.backtest.total_trades for wr in wrs)
    total_wins = sum(wr.backtest.wins for wr in wrs)
    total_losses = sum(wr.backtest.losses for wr in wrs)
    short_trades = sum(wr.short_count for wr in wrs)
    long_trades = sum(wr.long_count for wr in wrs)

    gross_profit = 0.0
    gross_loss = 0.0
    for wr in wrs:
        for t in wr.backtest.trades:
            if t.pnl_pct > 0:
                gross_profit += t.pnl_pct
            elif t.pnl_pct < 0:
                gross_loss += abs(t.pnl_pct)

    return CategorySummary(
        category=category,
        windows=len(wrs),
        total_pnl=sum(week_pnls),
        weekly_win_rate=positive / len(wrs) if wrs else 0.0,
        positive_weeks=positive,
        worst_week_pnl=min(week_pnls) if week_pnls else 0.0,
        best_week_pnl=max(week_pnls) if week_pnls else 0.0,
        total_trades=total_trades,
        short_trades=short_trades,
        long_trades=long_trades,
        trade_win_rate=(total_wins / (total_wins + total_losses))
        if (total_wins + total_losses) > 0
        else 0.0,
        profit_factor=(
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        ),
        sortino_ratio=_annualized_sortino_ratio(
            wrs,
            risk_free_rate_annual=risk_free_rate_annual,
        ),
    )


def _annualized_sortino_ratio(
    wrs: list[WindowResult],
    *,
    risk_free_rate_annual: float,
) -> float:
    if not wrs:
        return 0.0

    excess_returns: list[float] = []
    window_days: list[float] = []

    for wr in wrs:
        period_return = wr.backtest.total_pnl_pct / 100.0
        days = _window_days(wr.window)
        if days <= 0:
            continue
        period_risk_free = (1.0 + risk_free_rate_annual) ** (days / 365.25) - 1.0
        excess_returns.append(period_return - period_risk_free)
        window_days.append(days)

    if not excess_returns:
        return 0.0

    mean_excess = sum(excess_returns) / len(excess_returns)
    downside_variance = sum(min(excess, 0.0) ** 2 for excess in excess_returns) / len(excess_returns)
    downside_deviation = math.sqrt(downside_variance)
    if downside_deviation == 0.0:
        return float("inf") if mean_excess > 0.0 else 0.0

    avg_days = sum(window_days) / len(window_days)
    annualization = math.sqrt(365.25 / avg_days)
    return annualization * mean_excess / downside_deviation


def _window_days(window: EvalWindow) -> float:
    return (window.end - window.start).total_seconds() / 86400.0
