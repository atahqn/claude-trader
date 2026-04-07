"""Unified strategy evaluator.

Replaces one-off evaluation scripts with a reusable evaluator that takes
a :class:`SignalGenerator` and a set of :class:`EvalWindow` definitions.
"""

from __future__ import annotations

import csv
import json
import os
import random as _random_module
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .calibration import CalibrationResult, search_parameters, validate_calibration_config
from .engine import (
    DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS,
    _compute_stats,
    _process_batch,
    backtest_signals,
)
from .eval_windows import CATEGORY_DESCRIPTIONS, EvalWindow
from .models import BacktestResult, ExitReason, PositionType, Signal, TradeResult
from .pipeline import BacktestExecutionSession, PreparedMarketContext, prepare_market_context
from .preview import interval_to_timedelta

if TYPE_CHECKING:
    from live.signal_generator import CompositeSignalGenerator, SignalGenerator

    from .data import BinanceClient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_USD_RISK_FREE_RATE_ANNUAL = 0.0373
"""Annual USD risk-free proxy used for Sortino calculations.

Based on the U.S. Treasury 3-month constant maturity rate in the Federal
Reserve H.15 release dated March 27, 2026 (3.73% for March 26, 2026).
"""


_DATA_BUFFER_HOURS = 168
"""Hours of extra market data to fetch beyond the last signal window.

This ensures we have enough candles to resolve trades whose holding period
extends past the final evaluation window.  The actual holding time per trade
is governed by ``Signal.max_holding_hours``.
"""

_PREFERENCE_DRAWDOWN_FLOOR_PCT = 5.0
"""Floor on drawdown in the preference score to avoid fragile infinite ratios."""

_PREFERENCE_TRADE_SCALE = 80.0
"""Trade-count scale used in the preference coverage penalty."""


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    approximate: bool = True
    seed: int | None = None
    risk_free_rate_annual: float = DEFAULT_USD_RISK_FREE_RATE_ANNUAL
    entry_delay_seconds: int = DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS
    data_max_workers: int = 8
    backtest_max_workers: int = 0


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
    resolved_trades: int
    short_trades: int
    long_trades: int
    active_weeks: int
    trade_win_rate: float
    profit_factor: float
    sortino_ratio: float
    max_drawdown_pct: float
    pnl_to_mdd: float
    weekly_omega_ratio: float
    coverage_penalty: float
    preference_eligible: bool
    preference_score: float

    def preference_sort_key(self) -> tuple[float, float, float, float, float, float]:
        """Sort key for direct strategy ranking.

        Eligible strategies always outrank ineligible ones. Within each group,
        prefer the composite score first, then the stated tie-breakers.
        """
        return (
            1.0 if self.preference_eligible else 0.0,
            self.preference_score,
            _finite_or_cap(self.sortino_ratio),
            self.weekly_win_rate,
            _finite_or_cap(self.profit_factor),
            self.total_pnl,
        )


@dataclass(slots=True)
class SymbolSummary:
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    avg_hold_hours: float
    short_trades: int
    short_wins: int
    short_pnl: float
    long_trades: int
    long_wins: int
    long_pnl: float
    tp_exits: int
    sl_exits: int
    timeout_exits: int
    unfilled: int


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

    def resolved_trade_breakdown(self) -> tuple[int, int]:
        exact = 0
        fallback = 0
        for wr in self.window_results:
            for trade in wr.backtest.trades:
                if trade.exit_reason is ExitReason.UNFILLED:
                    continue
                if trade.used_fallback:
                    fallback += 1
                else:
                    exact += 1
        return exact, fallback

    def format_table(self) -> str:
        rows = self.all_summaries() + [self.overall_summary()]
        header = (
            f"{'Category':<35} | {'Win':>7} | {'PNL':>9} | {'WR':>6} "
            f"| {'Worst':>8} | {'Best':>8} | {'Trades':>6} "
            f"| {'S/L':>9} | {'Trd WR':>6} | {'PF':>5} | {'Sort':>7} "
            f"| {'DD':>6} | {'Omega':>6} | {'Pref':>7}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for s in rows:
            desc = CATEGORY_DESCRIPTIONS.get(s.category, "")
            label = f"{s.category} ({desc})" if desc else s.category
            lines.append(
                f"{label:<35} | {s.windows:>7} | "
                f"{s.total_pnl:>+8.2f}% | "
                f"{s.weekly_win_rate * 100:>5.1f}% | "
                f"{s.worst_week_pnl:>+7.2f}% | "
                f"{s.best_week_pnl:>+7.2f}% | "
                f"{s.total_trades:>6} | "
                f"{s.short_trades:>4}/{s.long_trades:<4} | "
                f"{s.trade_win_rate * 100:>5.1f}% | "
                f"{s.profit_factor:>5.2f} | "
                f"{s.sortino_ratio:>7.2f} | "
                f"{s.max_drawdown_pct:>5.1f}% | "
                f"{_format_metric(s.weekly_omega_ratio, width=6)} | "
                f"{_format_metric(s.preference_score, width=7)}"
            )
        return "\n".join(lines)

    def symbol_summaries(self) -> list[SymbolSummary]:
        """Compute per-symbol performance summaries from all trades."""
        raw: dict[str, dict] = defaultdict(lambda: {
            "trades": 0, "wins": 0, "losses": 0,
            "pnl": 0.0, "hold_hours": 0.0,
            "short_trades": 0, "short_wins": 0, "short_pnl": 0.0,
            "long_trades": 0, "long_wins": 0, "long_pnl": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "tp": 0, "sl": 0, "timeout": 0, "unfilled": 0,
        })

        for wr in self.window_results:
            for t in wr.backtest.trades:
                sym = t.signal.ticker
                s = raw[sym]
                weighted = t.pnl_pct * t.signal.size_multiplier

                s["trades"] += 1
                if t.exit_reason is ExitReason.UNFILLED:
                    s["unfilled"] += 1
                else:
                    hold_h = (t.exit_time - t.entry_time).total_seconds() / 3600
                    s["hold_hours"] += hold_h
                    if t.pnl_pct > 0:
                        s["wins"] += 1
                        s["gross_profit"] += weighted
                    elif t.pnl_pct < 0:
                        s["losses"] += 1
                        s["gross_loss"] += abs(weighted)

                if t.exit_reason is ExitReason.TP:
                    s["tp"] += 1
                elif t.exit_reason is ExitReason.SL:
                    s["sl"] += 1
                elif t.exit_reason is ExitReason.TIMEOUT:
                    s["timeout"] += 1

                s["pnl"] += weighted

                if t.signal.position_type is PositionType.SHORT:
                    s["short_trades"] += 1
                    s["short_pnl"] += weighted
                    if t.pnl_pct > 0:
                        s["short_wins"] += 1
                else:
                    s["long_trades"] += 1
                    s["long_pnl"] += weighted
                    if t.pnl_pct > 0:
                        s["long_wins"] += 1

        summaries = []
        for sym in sorted(raw, key=lambda k: raw[k]["pnl"], reverse=True):
            s = raw[sym]
            resolved = s["trades"] - s["unfilled"]
            decided = s["wins"] + s["losses"]
            gl = s["gross_loss"]
            gp = s["gross_profit"]
            summaries.append(SymbolSummary(
                symbol=sym,
                total_trades=s["trades"],
                wins=s["wins"],
                losses=s["losses"],
                win_rate=s["wins"] / decided if decided else 0.0,
                total_pnl=s["pnl"],
                avg_pnl=s["pnl"] / resolved if resolved else 0.0,
                profit_factor=(
                    gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
                ),
                avg_hold_hours=s["hold_hours"] / resolved if resolved else 0.0,
                short_trades=s["short_trades"],
                short_wins=s["short_wins"],
                short_pnl=s["short_pnl"],
                long_trades=s["long_trades"],
                long_wins=s["long_wins"],
                long_pnl=s["long_pnl"],
                tp_exits=s["tp"],
                sl_exits=s["sl"],
                timeout_exits=s["timeout"],
                unfilled=s["unfilled"],
            ))
        return summaries

    def save(self, output_dir: str | Path) -> Path:
        """Save full evaluation results to *output_dir*.

        Creates four files:
        - ``trades.csv``            — every trade with hold time, entry/exit
        - ``symbol_summary.csv``    — per-symbol aggregated stats
        - ``category_summary.csv``  — per-category aggregated stats
        - ``meta.json``             — config, symbols, timestamp
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._save_trades_csv(out / "trades.csv")
        self._save_symbol_summary_csv(out / "symbol_summary.csv")
        self._save_category_summary_csv(out / "category_summary.csv")
        self._save_meta_json(out / "meta.json")

        return out

    # -- private save helpers ------------------------------------------------

    def _save_trades_csv(self, path: Path) -> None:
        fields = [
            "window", "category", "symbol", "direction",
            "signal_date", "entry_time", "exit_time", "hold_hours",
            "entry_price", "exit_price", "tp_price", "sl_price",
            "tp_pct", "sl_pct", "exit_reason", "resolution_level", "used_fallback",
            "pnl_pct", "gross_pnl_pct", "fee_drag_pct",
            "leverage", "size_multiplier", "max_holding_hours",
            "metadata",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for wr in self.window_results:
                for t in wr.backtest.trades:
                    if t.exit_reason is ExitReason.UNFILLED:
                        hold_h = 0.0
                    else:
                        hold_h = (t.exit_time - t.entry_time).total_seconds() / 3600
                    writer.writerow({
                        "window": wr.window.name,
                        "category": wr.window.category,
                        "symbol": t.signal.ticker,
                        "direction": t.signal.position_type.value,
                        "signal_date": _iso(t.signal.signal_date),
                        "entry_time": _iso(t.entry_time),
                        "exit_time": _iso(t.exit_time),
                        "hold_hours": round(hold_h, 2),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "tp_price": t.tp_price,
                        "sl_price": t.sl_price,
                        "tp_pct": t.signal.tp_pct,
                        "sl_pct": t.signal.sl_pct,
                        "exit_reason": t.exit_reason.value,
                        "resolution_level": t.resolution_level.value,
                        "used_fallback": t.used_fallback,
                        "pnl_pct": round(t.pnl_pct, 4),
                        "gross_pnl_pct": round(t.gross_pnl_pct, 4),
                        "fee_drag_pct": round(t.fee_drag_pct, 4),
                        "leverage": t.signal.leverage,
                        "size_multiplier": t.signal.size_multiplier,
                        "max_holding_hours": t.signal.max_holding_hours,
                        "metadata": json.dumps(t.signal.metadata) if t.signal.metadata else "",
                    })

    def _save_symbol_summary_csv(self, path: Path) -> None:
        summaries = self.symbol_summaries()
        fields = [
            "symbol", "total_trades", "wins", "losses", "win_rate",
            "total_pnl", "avg_pnl", "profit_factor", "avg_hold_hours",
            "short_trades", "short_wins", "short_pnl",
            "long_trades", "long_wins", "long_pnl",
            "tp_exits", "sl_exits", "timeout_exits", "unfilled",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in summaries:
                writer.writerow({
                    "symbol": s.symbol,
                    "total_trades": s.total_trades,
                    "wins": s.wins,
                    "losses": s.losses,
                    "win_rate": round(s.win_rate, 4),
                    "total_pnl": round(s.total_pnl, 4),
                    "avg_pnl": round(s.avg_pnl, 4),
                    "profit_factor": round(s.profit_factor, 4)
                    if math.isfinite(s.profit_factor) else "inf",
                    "avg_hold_hours": round(s.avg_hold_hours, 2),
                    "short_trades": s.short_trades,
                    "short_wins": s.short_wins,
                    "short_pnl": round(s.short_pnl, 4),
                    "long_trades": s.long_trades,
                    "long_wins": s.long_wins,
                    "long_pnl": round(s.long_pnl, 4),
                    "tp_exits": s.tp_exits,
                    "sl_exits": s.sl_exits,
                    "timeout_exits": s.timeout_exits,
                    "unfilled": s.unfilled,
                })

    def _save_category_summary_csv(self, path: Path) -> None:
        rows = self.all_summaries() + [self.overall_summary()]
        fields = [
            "category", "windows", "total_pnl", "weekly_win_rate",
            "positive_weeks", "worst_week_pnl", "best_week_pnl",
            "total_trades", "resolved_trades", "short_trades", "long_trades",
            "active_weeks", "trade_win_rate", "profit_factor", "sortino_ratio",
            "max_drawdown_pct", "pnl_to_mdd", "weekly_omega_ratio",
            "coverage_penalty", "preference_eligible", "preference_score",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in rows:
                writer.writerow({
                    "category": s.category,
                    "windows": s.windows,
                    "total_pnl": round(s.total_pnl, 4),
                    "weekly_win_rate": round(s.weekly_win_rate, 4),
                    "positive_weeks": s.positive_weeks,
                    "worst_week_pnl": round(s.worst_week_pnl, 4),
                    "best_week_pnl": round(s.best_week_pnl, 4),
                    "total_trades": s.total_trades,
                    "resolved_trades": s.resolved_trades,
                    "short_trades": s.short_trades,
                    "long_trades": s.long_trades,
                    "active_weeks": s.active_weeks,
                    "trade_win_rate": round(s.trade_win_rate, 4),
                    "profit_factor": round(s.profit_factor, 4)
                    if math.isfinite(s.profit_factor) else "inf",
                    "sortino_ratio": round(s.sortino_ratio, 4)
                    if math.isfinite(s.sortino_ratio) else "inf",
                    "max_drawdown_pct": round(s.max_drawdown_pct, 4),
                    "pnl_to_mdd": round(s.pnl_to_mdd, 4)
                    if math.isfinite(s.pnl_to_mdd) else "inf",
                    "weekly_omega_ratio": round(s.weekly_omega_ratio, 4)
                    if math.isfinite(s.weekly_omega_ratio) else "inf",
                    "coverage_penalty": round(s.coverage_penalty, 4),
                    "preference_eligible": s.preference_eligible,
                    "preference_score": round(s.preference_score, 4)
                    if math.isfinite(s.preference_score) else "inf",
                })

    def _save_meta_json(self, path: Path) -> None:
        meta = {
            "saved_at": _iso(datetime.now(timezone.utc)),
            "symbols": self.symbols,
            "config": {
                "approximate": self.config.approximate,
                "seed": self.config.seed,
                "risk_free_rate_annual": self.config.risk_free_rate_annual,
                "entry_delay_seconds": self.config.entry_delay_seconds,
                "data_max_workers": self.config.data_max_workers,
                "backtest_max_workers": self.config.backtest_max_workers,
            },
            "total_windows": len(self.window_results),
            "windows": [
                {
                    "name": wr.window.name,
                    "category": wr.window.category,
                    "start": _iso(wr.window.start),
                    "end": _iso(wr.window.end),
                    "signal_count": wr.signal_count,
                    "short_count": wr.short_count,
                    "long_count": wr.long_count,
                    "total_pnl_pct": round(wr.backtest.total_pnl_pct, 4),
                    "trades": wr.backtest.total_trades,
                    "wins": wr.backtest.wins,
                    "losses": wr.backtest.losses,
                    "win_rate": round(wr.backtest.win_rate, 4),
                    "profit_factor": round(wr.backtest.profit_factor, 4)
                    if math.isfinite(wr.backtest.profit_factor) else "inf",
                    "max_drawdown_pct": round(wr.backtest.max_drawdown_pct, 4),
                }
                for wr in self.window_results
            ],
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """Format datetime as ISO 8601 string."""
    return dt.isoformat() if dt else ""


def _format_metric(value: float, *, width: int) -> str:
    if math.isinf(value):
        return f"{'inf':>{width}}"
    return f"{value:>{width}.2f}"


# ---------------------------------------------------------------------------
# Period-level parallel backtesting
# ---------------------------------------------------------------------------


def _resolve_signals_parallel(
    *,
    signals: list[Signal],
    approximate: bool,
    seed: int | None,
    entry_delay_seconds: int,
    max_workers: int,
    client: object = None,
) -> list[TradeResult]:
    """Resolve a flat list of signals in parallel using ProcessPoolExecutor.

    Each process creates its own BinanceClient (with shared disk cache).
    No PreparedMarketContext is passed — processes use the disk cache directly.
    Returns trades in the same order as the input signals.
    """
    if not signals:
        return []

    parent_rng = _random_module.Random(seed)
    signal_seeds = [parent_rng.getrandbits(64) for _ in signals]

    def _sequential() -> list[TradeResult]:
        c = client
        if c is None:
            from .data import BinanceClient
            c = BinanceClient()
        session = BacktestExecutionSession(client=c)
        from .engine import backtest_signal

        trades: list[TradeResult] = []
        for signal, sig_seed in zip(signals, signal_seeds):
            rng = _random_module.Random(sig_seed)
            trades.append(backtest_signal(
                signal, approximate=approximate, rng=rng,
                session=session, default_entry_delay_seconds=entry_delay_seconds,
            ))
        return trades

    effective_workers = (
        min(len(signals), max_workers or (os.cpu_count() or 4))
        if max_workers != 1
        else 1
    )

    if effective_workers <= 1:
        return _sequential()

    indexed = [
        (i, sig, sig_seed)
        for i, (sig, sig_seed) in enumerate(zip(signals, signal_seeds))
    ]
    batch_size = math.ceil(len(indexed) / effective_workers)
    batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]

    try:
        all_indexed: list[tuple[int, TradeResult]] = []
        with ProcessPoolExecutor(max_workers=len(batches)) as pool:
            futures = [
                pool.submit(
                    _process_batch, batch, approximate, entry_delay_seconds, None,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                all_indexed.extend(future.result())
        all_indexed.sort(key=lambda x: x[0])
        return [t for _, t in all_indexed]
    except (TypeError, AttributeError):
        return _sequential()


def _backtest_period(
    *,
    window_slices: list[tuple[EvalWindow, list[Signal], int, int]],
    ctx: PreparedMarketContext,
    approximate: bool,
    seed: int | None,
    entry_delay_seconds: int,
    max_workers: int,
    client: object,
) -> list[BacktestResult]:
    """Backtest all signals in a period, optionally using multiprocessing.

    Collects signals from all windows, batches them across worker processes,
    then re-distributes results to per-window BacktestResults.
    """
    import os
    import random as _random_module
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Flatten all window signals into one list with (global_idx, window_idx) tracking.
    all_signals: list[Signal] = []
    window_offsets: list[tuple[int, int]] = []  # (start_idx, count) per window
    for _, w_sigs, _, _ in window_slices:
        offset = len(all_signals)
        all_signals.extend(w_sigs)
        window_offsets.append((offset, len(w_sigs)))

    # Empty period — return empty results for each window.
    if not all_signals:
        return [
            BacktestResult(trades=[], **_compute_stats([]))
            for _ in window_slices
        ]

    # Derive deterministic per-signal seeds.
    parent_rng = _random_module.Random(seed)
    signal_seeds = [parent_rng.getrandbits(64) for _ in all_signals]

    effective_workers = (
        min(len(all_signals), max_workers or (os.cpu_count() or 4))
        if max_workers != 1
        else 1
    )

    if effective_workers <= 1:
        # Sequential path.
        session = BacktestExecutionSession(client=client, prepared_context=ctx)
        from .engine import backtest_signal

        all_trades: list[TradeResult] = []
        for signal, sig_seed in zip(all_signals, signal_seeds):
            rng = _random_module.Random(sig_seed)
            result = backtest_signal(
                signal,
                approximate=approximate,
                rng=rng,
                session=session,
                default_entry_delay_seconds=entry_delay_seconds,
            )
            all_trades.append(result)
    else:
        # Parallel path — batch across processes.
        indexed = [
            (i, sig, sig_seed)
            for i, (sig, sig_seed) in enumerate(zip(all_signals, signal_seeds))
        ]
        batch_size = math.ceil(len(indexed) / effective_workers)
        batches = [
            indexed[i : i + batch_size]
            for i in range(0, len(indexed), batch_size)
        ]

        try:
            all_indexed: list[tuple[int, TradeResult]] = []
            with ProcessPoolExecutor(max_workers=len(batches)) as pool:
                futures = [
                    pool.submit(
                        _process_batch,
                        batch,
                        approximate,
                        entry_delay_seconds,
                        ctx,
                    )
                    for batch in batches
                ]
                for future in as_completed(futures):
                    all_indexed.extend(future.result())
            all_indexed.sort(key=lambda x: x[0])
            all_trades = [t for _, t in all_indexed]
        except (TypeError, AttributeError):
            # Context or signals not picklable — fall back to sequential.
            session = BacktestExecutionSession(client=client, prepared_context=ctx)
            from .engine import backtest_signal

            all_trades = []
            for signal, sig_seed in zip(all_signals, signal_seeds):
                rng = _random_module.Random(sig_seed)
                all_trades.append(backtest_signal(
                    signal, approximate=approximate, rng=rng,
                    session=session,
                    default_entry_delay_seconds=entry_delay_seconds,
                ))


    # Re-distribute trades to per-window results.
    results: list[BacktestResult] = []
    for start_idx, count in window_offsets:
        window_trades = all_trades[start_idx : start_idx + count]
        window_trades.sort(key=lambda t: t.entry_time)
        results.append(BacktestResult(trades=window_trades, **_compute_stats(window_trades)))
    return results


# ---------------------------------------------------------------------------
# Cooldown enforcement across chunk/period boundaries
# ---------------------------------------------------------------------------


def _enforce_cooldown(
    signals: list[Signal],
    cooldown_hours: float,
) -> list[Signal]:
    """Remove signals that violate per-symbol cooldown.

    Expects *signals* sorted by ``signal_date``.  Keeps the first signal
    for each symbol and drops any subsequent signal within *cooldown_hours*
    of the previous kept signal for the same symbol.
    """
    if cooldown_hours <= 0:
        return signals
    last_by_symbol: dict[str, datetime] = {}
    kept: list[Signal] = []
    for sig in signals:
        sym = sig.ticker
        last = last_by_symbol.get(sym)
        if last is not None:
            hours = (sig.signal_date - last).total_seconds() / 3600.0
            if hours < cooldown_hours:
                continue
        last_by_symbol[sym] = sig.signal_date
        kept.append(sig)
    return kept


# ---------------------------------------------------------------------------
# Parallel signal generation worker (top-level for pickle)
# ---------------------------------------------------------------------------


# A strategy factory is either a single (module, class, kwargs) tuple or
# a list of them for composite strategies.
StrategyFactory = (
    tuple[str, str, dict[str, Any]]
    | list[tuple[str, str, dict[str, Any]]]
)


def _build_generator_from_factory(
    factory: StrategyFactory,
) -> Any:
    """Reconstruct a SignalGenerator from a factory spec."""
    import importlib

    if isinstance(factory, list):
        from live.signal_generator import CompositeSignalGenerator

        generators = [_build_generator_from_factory(f) for f in factory]
        return CompositeSignalGenerator(generators)

    module_name, class_name, kwargs = factory
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    return strategy_cls(**kwargs)


def _build_calibration_frame(
    ctx: PreparedMarketContext,
    symbols: list[str],
    lookback_start: datetime,
    lookback_end: datetime,
) -> pd.DataFrame | None:
    """Build a combined OHLCV frame for the calibration lookback window."""
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            smc = ctx.for_symbol(symbol)
        except KeyError:
            continue
        if smc.frame.empty:
            continue
        mask = (smc.frame["close_time"] >= lookback_start) & (smc.frame["close_time"] < lookback_end)
        sliced = smc.frame.loc[mask]
        if not sliced.empty:
            chunk = sliced.copy()
            chunk["symbol"] = symbol
            frames.append(chunk)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _calibrate_generator(
    generator: Any,
    ctx: PreparedMarketContext,
    symbols: list[str],
    chunk_start: datetime,
    calib_lookback: timedelta,
) -> None:
    """Run one calibration cycle and set ``active_params`` on *generator*."""
    from live.signal_generator import CompositeSignalGenerator

    lookback_start = chunk_start - calib_lookback

    if isinstance(generator, CompositeSignalGenerator):
        for child in generator.calibration_children():
            child_symbols = [s for s in symbols if s in child.symbols]
            _calibrate_single(child, ctx, child_symbols, lookback_start, chunk_start)
    else:
        _calibrate_single(generator, ctx, symbols, lookback_start, chunk_start)


def _calibrate_single(
    generator: Any,
    ctx: PreparedMarketContext,
    symbols: list[str],
    lookback_start: datetime,
    lookback_end: datetime,
) -> None:
    raw_frame = _build_calibration_frame(ctx, symbols, lookback_start, lookback_end)
    if raw_frame is None or raw_frame.empty:
        return
    frame = generator.build_calibration_frame(raw_frame, lookback_end)
    if frame is None or frame.empty:
        return
    result = search_parameters(
        param_space=generator.param_space(),
        score_fn=generator.score_params,
        frame=frame,
    )
    if result is not None:
        generator.active_params = result.best_params


def _generate_signals_with_calibration(
    generator: Any,
    ctx: PreparedMarketContext,
    symbols: list[str],
    sig_start: datetime,
    sig_end: datetime,
    calib_interval: timedelta,
    calib_lookback: timedelta,
    calibration_anchor: datetime | None = None,
) -> list[Signal]:
    """Generate signals with rolling recalibration.

    Calibration points are anchored to *calibration_anchor* (defaults to
    *sig_start*) so that parallel chunks sharing the same anchor produce
    identical calibration schedules.
    """
    if calibration_anchor is None:
        calibration_anchor = sig_start

    # Build deterministic calibration boundaries within [sig_start, sig_end).
    interval_secs = calib_interval.total_seconds()
    elapsed = (sig_start - calibration_anchor).total_seconds()
    k = max(0, int(elapsed // interval_secs))

    boundaries: list[datetime] = []
    t = calibration_anchor + timedelta(seconds=k * interval_secs)
    while t < sig_end:
        boundaries.append(t)
        t += calib_interval
    if not boundaries:
        boundaries = [sig_start]

    all_signals: list[Signal] = []
    for i, calib_time in enumerate(boundaries):
        _calibrate_generator(generator, ctx, symbols, calib_time, calib_lookback)

        gen_start = max(sig_start, calib_time)
        gen_end = boundaries[i + 1] if i + 1 < len(boundaries) else sig_end
        gen_end = min(gen_end, sig_end)

        if gen_start < gen_end:
            sigs = generator.generate_backtest_signals(ctx, symbols, gen_start, gen_end)
            all_signals.extend(sigs)

    all_signals.sort(key=lambda s: s.signal_date)
    return all_signals


def _generate_period_signals(
    period_idx: int,
    strategy_factory: StrategyFactory,
    symbols: list[str],
    signal_start: datetime,
    signal_end: datetime,
    fetch_end: datetime,
    request: Any,
    warmup_bars: int,
    data_max_workers: int,
    calibration_config: tuple[int, int, datetime] | None = None,
) -> tuple[int, PreparedMarketContext, list[Signal]]:
    """Worker: create a fresh strategy + client, generate signals for one period."""
    from .data import BinanceClient

    generator = _build_generator_from_factory(strategy_factory)

    client = BinanceClient()
    ctx = prepare_market_context(
        symbols, signal_start, fetch_end,
        client=client, request=request,
        warmup_bars=warmup_bars,
        max_workers=data_max_workers,
    )
    if calibration_config is not None and generator.needs_calibration:
        interval_h, lookback_h, anchor = calibration_config
        signals = _generate_signals_with_calibration(
            generator, ctx, symbols, signal_start, signal_end,
            calib_interval=timedelta(hours=interval_h),
            calib_lookback=timedelta(hours=lookback_h),
            calibration_anchor=anchor,
        )
    else:
        signals = generator.generate_backtest_signals(ctx, symbols, signal_start, signal_end)
    signals.sort(key=lambda s: s.signal_date)
    return period_idx, ctx, signals


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
        strategy_factory: tuple[str, str, dict[str, Any]] | None = None,
    ) -> None:
        self._symbols = symbols
        self._config = config
        self._cooldown_warmup = cooldown_warmup
        self._strategy_factory = strategy_factory
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

        request = generator.market_data_request()
        max_w = self._config.data_max_workers or (os.cpu_count() or 4)
        can_parallel = self._strategy_factory is not None and max_w > 1
        cooldown_h = generator.cooldown_hours

        uses_calibration = generator.needs_calibration
        calib_interval = timedelta(hours=generator.calibration_interval_hours)
        calib_lookback = timedelta(hours=generator.calibration_lookback_hours)
        if uses_calibration:
            validate_calibration_config(generator)

        # Warmup must cover calibration lookback when calibration is active.
        warmup_bars = 100
        if uses_calibration:
            interval_td = interval_to_timedelta(request.ohlcv_interval)
            interval_hours = interval_td.total_seconds() / 3600
            warmup_bars = max(warmup_bars, int(generator.calibration_lookback_hours / interval_hours))

        # ------------------------------------------------------------------
        # Build chunk list: subdivide periods so we have >= max_w chunks.
        # Each chunk: (chunk_idx, period_idx, windows, sig_start, sig_end,
        #              fetch_end)
        # ------------------------------------------------------------------
        period_params: list[tuple[list[EvalWindow], datetime, datetime, datetime]] = []
        for period_windows in periods:
            earliest_start = min(w.start for w in period_windows)
            latest_end = max(w.end for w in period_windows)
            signal_start = earliest_start - self._cooldown_warmup
            signal_end = latest_end
            fetch_end = signal_end + timedelta(hours=_DATA_BUFFER_HOURS)
            period_params.append((period_windows, signal_start, signal_end, fetch_end))

        chunks: list[tuple[int, int, list[EvalWindow], datetime, datetime, datetime]] = []
        chunk_idx = 0
        for period_idx, (pw, sig_start, sig_end, fetch_end) in enumerate(period_params):
            period_dur = sig_end - sig_start
            # Subdivide if parallel is available and the period is large
            # enough relative to the number of workers.
            _MIN_CHUNK_DUR = timedelta(days=7)
            n_chunks = 1
            if can_parallel and len(periods) < max_w:
                # Subdivide so each chunk is at least 1 week.
                n_chunks = min(
                    max(1, max_w // max(len(periods), 1)),
                    max(1, int(period_dur / _MIN_CHUNK_DUR)),
                )
            if n_chunks > 1:
                chunk_dur = period_dur / n_chunks
                for ci in range(n_chunks):
                    c_sig_start = sig_start + chunk_dur * ci
                    c_sig_end = sig_start + chunk_dur * (ci + 1)
                    if ci == n_chunks - 1:
                        c_sig_end = sig_end  # avoid float drift
                    c_fetch_end = c_sig_end + timedelta(hours=_DATA_BUFFER_HOURS)
                    chunks.append((chunk_idx, period_idx, pw, c_sig_start, c_sig_end, c_fetch_end))
                    chunk_idx += 1
            else:
                chunks.append((chunk_idx, period_idx, pw, sig_start, sig_end, fetch_end))
                chunk_idx += 1

        # ------------------------------------------------------------------
        # Phase 1: Signal generation — parallel across chunks.
        # ------------------------------------------------------------------
        # Result per chunk: (chunk_idx, period_idx, ctx, signals)
        chunk_results: list[tuple[int, int, PreparedMarketContext, list[Signal]]]

        if not can_parallel or len(chunks) <= 1:
            chunk_results = []
            for ci, period_idx, pw, sig_start, sig_end, fetch_end in chunks:
                ctx = prepare_market_context(
                    self._symbols, sig_start, fetch_end,
                    client=self._client, request=request,
                    warmup_bars=warmup_bars, max_workers=self._config.data_max_workers,
                )
                if uses_calibration:
                    anchor = period_params[period_idx][1]
                    sigs = _generate_signals_with_calibration(
                        generator, ctx, self._symbols, sig_start, sig_end,
                        calib_interval, calib_lookback,
                        calibration_anchor=anchor,
                    )
                else:
                    sigs = generator.generate_backtest_signals(
                        ctx, self._symbols, sig_start, sig_end,
                    )
                sigs.sort(key=lambda s: s.signal_date)
                chunk_results.append((ci, period_idx, ctx, sigs))
        else:
            try:
                chunk_results = []
                effective_w = min(len(chunks), max_w)
                with ProcessPoolExecutor(max_workers=effective_w) as pool:
                    futures = {
                        pool.submit(
                            _generate_period_signals,
                            ci,
                            self._strategy_factory,
                            list(self._symbols),
                            sig_start, sig_end, fetch_end,
                            request, warmup_bars,
                            1,  # data_max_workers=1 inside each process
                            (
                                generator.calibration_interval_hours,
                                generator.calibration_lookback_hours,
                                period_params[period_idx][1],
                            ) if uses_calibration else None,
                        ): (ci, period_idx)
                        for ci, period_idx, _, sig_start, sig_end, fetch_end in chunks
                    }
                    for future in as_completed(futures):
                        ci, period_idx = futures[future]
                        result_ci, ctx, sigs = future.result()
                        chunk_results.append((ci, period_idx, ctx, sigs))
                chunk_results.sort(key=lambda x: x[0])
            except (TypeError, AttributeError):
                chunk_results = []
                for ci, period_idx, pw, sig_start, sig_end, fetch_end in chunks:
                    ctx = prepare_market_context(
                        self._symbols, sig_start, fetch_end,
                        client=self._client, request=request,
                        warmup_bars=warmup_bars, max_workers=self._config.data_max_workers,
                    )
                    if uses_calibration:
                        anchor = period_params[period_idx][1]
                        sigs = _generate_signals_with_calibration(
                            generator, ctx, self._symbols, sig_start, sig_end,
                            calib_interval, calib_lookback,
                            calibration_anchor=anchor,
                        )
                    else:
                        sigs = generator.generate_backtest_signals(
                            ctx, self._symbols, sig_start, sig_end,
                        )
                    sigs.sort(key=lambda s: s.signal_date)
                    chunk_results.append((ci, period_idx, ctx, sigs))

        # ------------------------------------------------------------------
        # Merge chunks per period and enforce cooldown.
        # ------------------------------------------------------------------
        period_signals: dict[int, list[Signal]] = defaultdict(list)
        for _, period_idx, _, sigs in chunk_results:
            period_signals[period_idx].extend(sigs)

        for period_idx in period_signals:
            merged = sorted(period_signals[period_idx], key=lambda s: s.signal_date)
            period_signals[period_idx] = _enforce_cooldown(merged, cooldown_h)

        # ------------------------------------------------------------------
        # Phase 2: Backtest resolution — batch ALL signals, resolve in parallel.
        # ------------------------------------------------------------------
        all_signals_flat: list[Signal] = []
        window_info: list[tuple[EvalWindow, int, int, int, int]] = []

        for period_idx, (pw, _, _, _) in enumerate(period_params):
            sigs = period_signals.get(period_idx, [])
            for window in pw:
                lo = bisect_left(sigs, window.start, key=lambda s: s.signal_date)
                hi = bisect_left(sigs, window.end, key=lambda s: s.signal_date)
                w_sigs = sigs[lo:hi]
                short_count = sum(
                    1 for s in w_sigs if s.position_type is PositionType.SHORT
                )
                long_count = len(w_sigs) - short_count
                offset = len(all_signals_flat)
                all_signals_flat.extend(w_sigs)
                window_info.append((window, short_count, long_count, offset, len(w_sigs)))

        if all_signals_flat:
            all_trades = _resolve_signals_parallel(
                signals=all_signals_flat,
                approximate=self._config.approximate,
                seed=self._config.seed,
                entry_delay_seconds=self._config.entry_delay_seconds,
                max_workers=self._config.backtest_max_workers,
                client=self._client,
            )
        else:
            all_trades = []

        # Re-distribute trades to per-window results.
        all_window_results: list[WindowResult] = []
        for window, short_count, long_count, offset, count in window_info:
            window_trades = all_trades[offset : offset + count]
            window_trades.sort(key=lambda t: t.entry_time)
            stats = _compute_stats(window_trades)
            all_window_results.append(
                WindowResult(
                    window=window,
                    backtest=BacktestResult(trades=window_trades, **stats),
                    signal_count=len(window_trades),
                    short_count=short_count,
                    long_count=long_count,
                )
            )

        return EvaluationReport(
            window_results=all_window_results,
            config=self._config,
            symbols=list(self._symbols),
        )


def rank_evaluation_reports(
    reports: list[tuple[str, EvaluationReport]],
    *,
    category: str = "ALL",
) -> list[tuple[str, CategorySummary]]:
    """Rank multiple evaluated strategies by the shared preference system."""
    ranked: list[tuple[str, CategorySummary]] = []
    for label, report in reports:
        summary = report.overall_summary() if category == "ALL" else report.category_summary(category)
        ranked.append((label, summary))
    return sorted(ranked, key=lambda item: item[1].preference_sort_key(), reverse=True)


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
            resolved_trades=0,
            short_trades=0,
            long_trades=0,
            active_weeks=0,
            trade_win_rate=0.0,
            profit_factor=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            pnl_to_mdd=0.0,
            weekly_omega_ratio=0.0,
            coverage_penalty=0.0,
            preference_eligible=False,
            preference_score=0.0,
        )

    week_pnls = [wr.backtest.total_pnl_pct for wr in wrs]
    positive = sum(1 for p in week_pnls if p > 0)
    total_trades = sum(wr.backtest.total_trades for wr in wrs)
    resolved_trades = sum(_resolved_trade_count(wr.backtest) for wr in wrs)
    total_wins = sum(wr.backtest.wins for wr in wrs)
    total_losses = sum(wr.backtest.losses for wr in wrs)
    short_trades = sum(wr.short_count for wr in wrs)
    long_trades = sum(wr.long_count for wr in wrs)
    active_weeks = sum(1 for wr in wrs if _resolved_trade_count(wr.backtest) > 0)
    weighted_returns = _resolved_weighted_trade_returns(wrs)

    gross_profit = 0.0
    gross_loss = 0.0
    for wr in wrs:
        for t in wr.backtest.trades:
            weighted = t.pnl_pct * t.signal.size_multiplier
            if weighted > 0:
                gross_profit += weighted
            elif weighted < 0:
                gross_loss += abs(weighted)

    max_drawdown_pct = _max_drawdown_pct(weighted_returns)
    pnl_to_mdd = (
        sum(week_pnls) / max_drawdown_pct
        if max_drawdown_pct > 0.0
        else (float("inf") if sum(week_pnls) > 0.0 else 0.0)
    )
    weekly_omega_ratio = _weekly_omega_ratio(week_pnls)
    coverage_penalty = _preference_coverage_penalty(
        active_weeks=active_weeks,
        total_weeks=len(wrs),
        resolved_trades=resolved_trades,
    )
    preference_eligible = _preference_eligible(
        total_pnl=sum(week_pnls),
        profit_factor=(
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        ),
        resolved_trades=resolved_trades,
        active_weeks=active_weeks,
        total_weeks=len(wrs),
    )
    preference_score = _preference_score(
        week_pnls=week_pnls,
        total_pnl=sum(week_pnls),
        max_drawdown_pct=max_drawdown_pct,
        coverage_penalty=coverage_penalty,
    )

    return CategorySummary(
        category=category,
        windows=len(wrs),
        total_pnl=sum(week_pnls),
        weekly_win_rate=positive / len(wrs) if wrs else 0.0,
        positive_weeks=positive,
        worst_week_pnl=min(week_pnls) if week_pnls else 0.0,
        best_week_pnl=max(week_pnls) if week_pnls else 0.0,
        total_trades=total_trades,
        resolved_trades=resolved_trades,
        short_trades=short_trades,
        long_trades=long_trades,
        active_weeks=active_weeks,
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
        max_drawdown_pct=max_drawdown_pct,
        pnl_to_mdd=pnl_to_mdd,
        weekly_omega_ratio=weekly_omega_ratio,
        coverage_penalty=coverage_penalty,
        preference_eligible=preference_eligible,
        preference_score=preference_score,
    )


def _resolved_trade_count(backtest: BacktestResult) -> int:
    return sum(1 for t in backtest.trades if t.exit_reason is not ExitReason.UNFILLED)


def _resolved_weighted_trade_returns(wrs: list[WindowResult]) -> list[float]:
    returns: list[float] = []
    for wr in wrs:
        for trade in wr.backtest.trades:
            if trade.exit_reason is ExitReason.UNFILLED:
                continue
            returns.append(trade.pnl_pct * trade.signal.size_multiplier)
    return returns


def _max_drawdown_pct(weighted_returns: list[float]) -> float:
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for ret in weighted_returns:
        equity *= 1.0 + ret / 100.0
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100.0
        max_dd = max(max_dd, dd)
    return max_dd


def _weekly_omega_ratio(week_pnls: list[float]) -> float:
    gross_positive = sum(max(pnl, 0.0) for pnl in week_pnls)
    gross_negative = sum(max(-pnl, 0.0) for pnl in week_pnls)
    if gross_negative > 0.0:
        return gross_positive / gross_negative
    if gross_positive > 0.0:
        return float("inf")
    return 0.0


def _preference_coverage_penalty(
    *,
    active_weeks: int,
    total_weeks: int,
    resolved_trades: int,
) -> float:
    if total_weeks <= 0:
        return 0.0
    week_component = math.sqrt(active_weeks / total_weeks)
    trade_component = min(1.0, resolved_trades / _PREFERENCE_TRADE_SCALE)
    return week_component * trade_component


def _preference_eligible(
    *,
    total_pnl: float,
    profit_factor: float,
    resolved_trades: int,
    active_weeks: int,
    total_weeks: int,
) -> bool:
    min_resolved_trades = min(40, max(10, 2 * total_weeks))
    min_active_weeks = min(8, total_weeks)
    return (
        total_pnl > 0.0
        and profit_factor > 1.0
        and resolved_trades >= min_resolved_trades
        and active_weeks >= min_active_weeks
    )


def _preference_score(
    *,
    week_pnls: list[float],
    total_pnl: float,
    max_drawdown_pct: float,
    coverage_penalty: float,
) -> float:
    if coverage_penalty <= 0.0 or total_pnl <= 0.0:
        return 0.0

    gross_positive = sum(max(pnl, 0.0) for pnl in week_pnls)
    gross_negative = sum(max(-pnl, 0.0) for pnl in week_pnls)
    omega_component = gross_positive / max(gross_negative, 1.0)
    drawdown_component = total_pnl / max(max_drawdown_pct, _PREFERENCE_DRAWDOWN_FLOOR_PCT)
    return coverage_penalty * omega_component * drawdown_component


def _finite_or_cap(value: float, *, cap: float = 1e12) -> float:
    if math.isnan(value):
        return float("-inf")
    if math.isinf(value):
        return cap if value > 0 else -cap
    return value


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
