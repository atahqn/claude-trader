from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta

from backtester import (
    ALL_WINDOWS,
    DEVELOPMENT_WINDOWS,
    EVALUATION_WINDOWS,
    StrategyEvaluator,
    validate_no_lookahead,
)
from backtester.evaluator import PortfolioConfig
from backtester.models import ExitReason, PositionType, ResolutionLevel, TradeResult
from backtester.pipeline import prepare_market_context
from live.squeeze_v8_strategy import SYMBOLS, SqueezeV8Strategy


def _window_set(name: str):
    if name == "development":
        return DEVELOPMENT_WINDOWS
    if name == "evaluation":
        return EVALUATION_WINDOWS
    return ALL_WINDOWS


def _window_label(name: str) -> str:
    return {
        "development": "Development",
        "evaluation": "Evaluation",
        "all": "Full Calendar",
    }[name]


def _max_consecutive_losses(trades: list[TradeResult]) -> int:
    best = 0
    current = 0
    for trade in trades:
        if trade.exit_reason is ExitReason.UNFILLED:
            continue
        if trade.pnl_pct < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_trades_per_day(trades: list[TradeResult]) -> tuple[str, int]:
    if not trades:
        return "-", 0
    counts = Counter(trade.entry_time.date().isoformat() for trade in trades)
    return max(counts.items(), key=lambda item: (item[1], item[0]))


def _max_losing_trades_per_day(trades: list[TradeResult]) -> tuple[str, int]:
    losses = [trade for trade in trades if trade.exit_reason is not ExitReason.UNFILLED and trade.pnl_pct < 0]
    if not losses:
        return "-", 0
    counts = Counter(trade.entry_time.date().isoformat() for trade in losses)
    return max(counts.items(), key=lambda item: (item[1], item[0]))


def _max_stop_losses_per_day(trades: list[TradeResult]) -> tuple[str, int]:
    stops = [trade for trade in trades if trade.exit_reason is ExitReason.SL]
    if not stops:
        return "-", 0
    counts = Counter(trade.entry_time.date().isoformat() for trade in stops)
    return max(counts.items(), key=lambda item: (item[1], item[0]))


def _format_metric(value: float, *, pct: bool = False) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    suffix = "%" if pct else ""
    return f"{value:.2f}{suffix}"


def _safe_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return 0.0


def _max_drawdown_from_trades(trades: list[TradeResult]) -> float:
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for trade in trades:
        if trade.exit_reason is ExitReason.UNFILLED:
            continue
        equity *= 1.0 + (trade.pnl_pct * trade.signal.size_multiplier) / 100.0
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return max_dd


@dataclass(slots=True)
class SideStats:
    label: str
    trades: int
    resolved_trades: int
    wins: int
    losses: int
    unfilled: int
    tp_exits: int
    sl_exits: int
    timeout_exits: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    profit_factor: float
    avg_hold_hours: float
    avg_entry_delay_seconds: float
    max_consecutive_losses: int


def _build_side_stats(label: str, trades: list[TradeResult]) -> SideStats:
    resolved = [trade for trade in trades if trade.exit_reason is not ExitReason.UNFILLED]
    wins = [trade for trade in resolved if trade.pnl_pct > 0]
    losses = [trade for trade in resolved if trade.pnl_pct < 0]
    weighted_returns = [trade.pnl_pct * trade.signal.size_multiplier for trade in resolved]

    gross_profit = sum(ret for ret in weighted_returns if ret > 0)
    gross_loss = abs(sum(ret for ret in weighted_returns if ret < 0))
    total_pnl = sum(weighted_returns)
    decided = len(wins) + len(losses)
    hold_hours = [
        (trade.exit_time - trade.entry_time).total_seconds() / 3600.0
        for trade in resolved
    ]
    entry_delays = [
        max((trade.entry_time - trade.signal.signal_date).total_seconds(), 0.0)
        for trade in trades
        if trade.exit_reason is not ExitReason.UNFILLED
    ]
    return SideStats(
        label=label,
        trades=len(trades),
        resolved_trades=len(resolved),
        wins=len(wins),
        losses=len(losses),
        unfilled=sum(1 for trade in trades if trade.exit_reason is ExitReason.UNFILLED),
        tp_exits=sum(1 for trade in resolved if trade.exit_reason is ExitReason.TP),
        sl_exits=sum(1 for trade in resolved if trade.exit_reason is ExitReason.SL),
        timeout_exits=sum(1 for trade in resolved if trade.exit_reason is ExitReason.TIMEOUT),
        total_pnl=total_pnl,
        avg_pnl=(total_pnl / len(resolved)) if resolved else 0.0,
        win_rate=(len(wins) / decided * 100.0) if decided else 0.0,
        profit_factor=_safe_profit_factor(gross_profit, gross_loss),
        avg_hold_hours=(sum(hold_hours) / len(hold_hours)) if hold_hours else 0.0,
        avg_entry_delay_seconds=(sum(entry_delays) / len(entry_delays)) if entry_delays else 0.0,
        max_consecutive_losses=_max_consecutive_losses(trades),
    )


def _side_table(stats: list[SideStats]) -> str:
    header = (
        f"{'Side':<8} | {'Trades':>6} | {'Win':>6} | {'Loss':>6} | {'Unfld':>5} "
        f"| {'PNL':>9} | {'Avg':>8} | {'WR':>6} | {'PF':>6} "
        f"| {'TP/SL/TO':>10} | {'Hold':>6} | {'Delay':>7} | {'MaxL':>4}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for stat in stats:
        rows.append(
            f"{stat.label:<8} | "
            f"{stat.trades:>6} | "
            f"{stat.wins:>6} | "
            f"{stat.losses:>6} | "
            f"{stat.unfilled:>5} | "
            f"{stat.total_pnl:>+8.2f}% | "
            f"{stat.avg_pnl:>7.2f}% | "
            f"{stat.win_rate:>5.1f}% | "
            f"{_format_metric(stat.profit_factor):>6} | "
            f"{stat.tp_exits:>2}/{stat.sl_exits:<2}/{stat.timeout_exits:<2} | "
            f"{stat.avg_hold_hours:>5.1f}h | "
            f"{stat.avg_entry_delay_seconds:>6.1f}s | "
            f"{stat.max_consecutive_losses:>4}"
        )
    return "\n".join(rows)


def _category_side_table(report) -> str:
    rows: list[str] = []
    header = (
        f"{'Category':<20} | {'Side':<5} | {'Trades':>6} | {'PNL':>9} | {'WR':>6} "
        f"| {'PF':>6} | {'TP':>4} | {'SL':>4} | {'TO':>4}"
    )
    sep = "-" * len(header)
    rows.extend([header, sep])

    by_category: dict[str, dict[PositionType, list[TradeResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for wr in report.window_results:
        for trade in wr.backtest.trades:
            by_category[wr.window.category][trade.signal.position_type].append(trade)

    for category in sorted(by_category):
        for side in (PositionType.SHORT, PositionType.LONG):
            side_trades = by_category[category].get(side, [])
            if not side_trades:
                continue
            stats = _build_side_stats(side.value, side_trades)
            rows.append(
                f"{category:<20} | {side.value:<5} | "
                f"{stats.trades:>6} | "
                f"{stats.total_pnl:>+8.2f}% | "
                f"{stats.win_rate:>5.1f}% | "
                f"{_format_metric(stats.profit_factor):>6} | "
                f"{stats.tp_exits:>4} | "
                f"{stats.sl_exits:>4} | "
                f"{stats.timeout_exits:>4}"
            )
    return "\n".join(rows)


def _resolution_table(trades: list[TradeResult]) -> str:
    counts = Counter(trade.resolution_level for trade in trades if trade.exit_reason is not ExitReason.UNFILLED)
    total = sum(counts.values()) or 1
    header = f"{'Resolution':<10} | {'Trades':>6} | {'Share':>6}"
    sep = "-" * len(header)
    rows = [header, sep]
    for level in (ResolutionLevel.TRADE, ResolutionLevel.MINUTE, ResolutionLevel.HOUR):
        count = counts.get(level, 0)
        share = count / total * 100.0
        rows.append(f"{level.value:<10} | {count:>6} | {share:>5.1f}%")
    return "\n".join(rows)


def _top_daily_rows(trades: list[TradeResult], *, limit: int = 8) -> str:
    per_day: dict[str, dict[str, float]] = defaultdict(
        lambda: {"trades": 0, "wins": 0, "losses": 0, "sl": 0, "pnl": 0.0}
    )
    for trade in trades:
        if trade.exit_reason is ExitReason.UNFILLED:
            continue
        key = trade.entry_time.date().isoformat()
        row = per_day[key]
        row["trades"] += 1
        row["pnl"] += trade.pnl_pct * trade.signal.size_multiplier
        if trade.pnl_pct > 0:
            row["wins"] += 1
        elif trade.pnl_pct < 0:
            row["losses"] += 1
        if trade.exit_reason is ExitReason.SL:
            row["sl"] += 1

    ranked = sorted(
        per_day.items(),
        key=lambda item: (item[1]["sl"], item[1]["losses"], -item[1]["pnl"], item[0]),
        reverse=True,
    )[:limit]
    header = f"{'Day':<12} | {'Trades':>6} | {'Loss':>6} | {'SL':>4} | {'PNL':>9}"
    sep = "-" * len(header)
    rows = [header, sep]
    for day, data in ranked:
        rows.append(
            f"{day:<12} | "
            f"{int(data['trades']):>6} | "
            f"{int(data['losses']):>6} | "
            f"{int(data['sl']):>4} | "
            f"{data['pnl']:>+8.2f}%"
        )
    return "\n".join(rows)


def _apply_daily_limits(
    trades: list[TradeResult],
    *,
    max_sl_per_day: int | None = None,
    max_entries_per_day: int | None = None,
) -> list[TradeResult]:
    if max_sl_per_day is None and max_entries_per_day is None:
        return list(trades)

    accepted: list[TradeResult] = []
    entries_taken: Counter[str] = Counter()

    for trade in sorted(trades, key=lambda item: item.entry_time):
        day = trade.entry_time.date().isoformat()

        if max_entries_per_day is not None and entries_taken[day] >= max_entries_per_day:
            continue

        if max_sl_per_day is not None:
            prior_same_day_stops = sum(
                1
                for accepted_trade in accepted
                if accepted_trade.exit_reason is ExitReason.SL
                and accepted_trade.exit_time.date().isoformat() == day
                and accepted_trade.exit_time <= trade.entry_time
            )
            if prior_same_day_stops >= max_sl_per_day:
                continue

        accepted.append(trade)
        entries_taken[day] += 1

    return accepted


def _risk_overlay_summary(
    trades: list[TradeResult],
    *,
    max_sl_per_day: int | None = None,
    max_entries_per_day: int | None = None,
) -> str:
    filtered = _apply_daily_limits(
        trades,
        max_sl_per_day=max_sl_per_day,
        max_entries_per_day=max_entries_per_day,
    )
    skipped = len(trades) - len(filtered)
    short_trades = [trade for trade in filtered if trade.signal.position_type is PositionType.SHORT]
    long_trades = [trade for trade in filtered if trade.signal.position_type is PositionType.LONG]
    total = _build_side_stats("ALL", filtered)

    lines = [
        "Risk Overlay",
        (
            f"Rules: max_sl_per_day={max_sl_per_day if max_sl_per_day is not None else '-'}, "
            f"max_entries_per_day={max_entries_per_day if max_entries_per_day is not None else '-'}"
        ),
        f"Accepted trades: {len(filtered)} / {len(trades)} (skipped {skipped})",
        f"Overlay PnL: {total.total_pnl:+.2f}%",
        f"Overlay PF: {_format_metric(total.profit_factor)}",
        f"Overlay WR: {total.win_rate:.1f}%",
        f"Overlay MDD: {_max_drawdown_from_trades(filtered):.2f}%",
        f"Overlay max consecutive losses: {total.max_consecutive_losses}",
        _side_table([
            _build_side_stats("SHORT", short_trades),
            _build_side_stats("LONG", long_trades),
            total,
        ]),
    ]
    return "\n".join(lines)


def _symbol_table(report, *, limit: int = 10) -> str:
    summaries = report.symbol_summaries()[:limit]
    header = (
        f"{'Symbol':<12} | {'Trades':>6} | {'PNL':>9} | {'WR':>6} | {'PF':>6} "
        f"| {'Short':>12} | {'Long':>12}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for summary in summaries:
        rows.append(
            f"{summary.symbol:<12} | "
            f"{summary.total_trades:>6} | "
            f"{summary.total_pnl:>+8.2f}% | "
            f"{summary.win_rate * 100:>5.1f}% | "
            f"{_format_metric(summary.profit_factor):>6} | "
            f"{summary.short_trades:>3}/{summary.short_pnl:>+7.2f}% | "
            f"{summary.long_trades:>3}/{summary.long_pnl:>+7.2f}%"
        )
    return "\n".join(rows)


def _run_lookahead_validation(strategy: SqueezeV8Strategy, windows, *, sample_size: int, seed: int) -> str:
    signal_start = min(window.start for window in windows) - timedelta(days=14)
    signal_end = max(window.end for window in windows)
    ctx = prepare_market_context(
        SYMBOLS,
        signal_start,
        signal_end,
        request=strategy.market_data_request(),
        warmup_bars=strategy.warmup_bars,
    )
    violations = validate_no_lookahead(
        strategy,
        ctx,
        SYMBOLS,
        signal_start,
        signal_end,
        sample_size=sample_size,
        seed=seed,
    )
    lines = [
        f"Look-ahead validation sample: {sample_size}",
        f"Look-ahead violations: {len(violations)}",
    ]
    for violation in violations[:10]:
        lines.append(f"- {violation.detail}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Squeeze V8.2 on the shared window calendar")
    parser.add_argument(
        "--windows",
        choices=("development", "evaluation", "all"),
        default="all",
        help="Which canonical window pack to evaluate",
    )
    parser.add_argument(
        "--sizing-mode",
        choices=("ridge_v1", "baseline"),
        default="ridge_v1",
        help="Signal size model to evaluate",
    )
    parser.add_argument(
        "--analysis-interval",
        choices=("1h", "30m", "15m", "5m", "1m"),
        default="1h",
        help="Base analysis candle interval for the strategy",
    )
    parser.add_argument(
        "--poll-interval",
        choices=("1h", "30m", "15m", "5m", "1m"),
        default=None,
        help="Optional preview poll interval. Must divide the analysis interval exactly.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default=None,
        help="Optional exact window name filter inside the selected pack (for example Apr25_W1)",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact trade-level exit resolution instead of approximate mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by approximate backtests and validation sampling",
    )
    parser.add_argument(
        "--lookahead-sample-size",
        type=int,
        default=0,
        help="Run truncation-based look-ahead validation on N sampled signals",
    )
    parser.add_argument(
        "--top-symbols",
        type=int,
        default=8,
        help="How many symbols to show in the symbol leaderboard",
    )
    parser.add_argument(
        "--max-sl-per-day",
        type=int,
        default=None,
        help="Experimental risk overlay: stop taking new trades on a day after N same-day stop exits",
    )
    parser.add_argument(
        "--max-entries-per-day",
        type=int,
        default=None,
        help="Experimental risk overlay: cap accepted entries per day",
    )
    args = parser.parse_args()

    windows = _window_set(args.windows)
    if args.window_name is not None:
        filtered = [window for window in windows if window.name == args.window_name]
        if not filtered:
            raise ValueError(f"window {args.window_name!r} was not found inside pack {args.windows!r}")
        windows = filtered
    sizing_mode = args.sizing_mode
    if args.analysis_interval != "1h" and sizing_mode == "ridge_v1":
        print(
            "ridge_v1 is calibrated only for 1h candles; falling back to baseline sizing "
            f"for analysis_interval={args.analysis_interval}",
        )
        sizing_mode = "baseline"

    strategy = SqueezeV8Strategy(
        analysis_interval=args.analysis_interval,
        poll_interval=args.poll_interval or args.analysis_interval,
        sizing_mode=sizing_mode,
    )
    config = PortfolioConfig(
        approximate=not args.exact,
        seed=args.seed,
    )
    report = StrategyEvaluator(
        SYMBOLS,
        config=config,
    ).evaluate(strategy, windows)

    print(report.format_table())

    all_trades: list[TradeResult] = []
    for wr in report.window_results:
        all_trades.extend(wr.backtest.trades)
    all_trades.sort(key=lambda trade: trade.entry_time)

    short_trades = [trade for trade in all_trades if trade.signal.position_type is PositionType.SHORT]
    long_trades = [trade for trade in all_trades if trade.signal.position_type is PositionType.LONG]

    overall = report.overall_summary()
    max_trade_day, max_trade_count = _max_trades_per_day(all_trades)
    max_loss_day, max_loss_count = _max_losing_trades_per_day(all_trades)
    max_sl_day, max_sl_count = _max_stop_losses_per_day(all_trades)

    print()
    print(f"Window pack: {_window_label(args.windows)}")
    print(f"Analysis interval: {args.analysis_interval}")
    print(f"Poll interval: {strategy.effective_poll_interval}")
    print(f"Sizing mode: {sizing_mode}")
    print(f"Execution realism: {'exact' if args.exact else 'approximate'}")
    print(f"Total trades: {overall.total_trades}")
    print(f"Resolved trades: {overall.resolved_trades}")
    print(f"Trade win rate: {overall.trade_win_rate * 100:.1f}%")
    print(f"Total PnL: {overall.total_pnl:+.2f}%")
    print(f"Profit factor: {overall.profit_factor:.2f}")
    print(f"Max drawdown: {overall.max_drawdown_pct:.2f}%")
    print(f"Max trades in a day: {max_trade_count} ({max_trade_day})")
    print(f"Max losing trades in a day: {max_loss_count} ({max_loss_day})")
    print(f"Max stop-loss exits in a day: {max_sl_count} ({max_sl_day})")
    print(f"Max consecutive losses: {_max_consecutive_losses(all_trades)}")

    print()
    print("Side Breakdown")
    print(_side_table([
        _build_side_stats("SHORT", short_trades),
        _build_side_stats("LONG", long_trades),
        _build_side_stats("ALL", all_trades),
    ]))

    print()
    print("Category x Side")
    print(_category_side_table(report))

    print()
    print("Exit Resolution")
    print(_resolution_table(all_trades))

    print()
    print("Worst Clustered Days")
    print(_top_daily_rows(all_trades))

    print()
    print("Top Symbols")
    print(_symbol_table(report, limit=args.top_symbols))

    if args.max_sl_per_day is not None or args.max_entries_per_day is not None:
        print()
        print(
            _risk_overlay_summary(
                all_trades,
                max_sl_per_day=args.max_sl_per_day,
                max_entries_per_day=args.max_entries_per_day,
            )
        )

    if args.lookahead_sample_size > 0:
        print()
        print("Look-ahead Check")
        print(
            _run_lookahead_validation(
                strategy,
                windows,
                sample_size=args.lookahead_sample_size,
                seed=args.seed,
            )
        )


if __name__ == "__main__":
    main()
