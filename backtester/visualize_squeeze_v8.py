from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from backtester import ALL_WINDOWS, DEVELOPMENT_WINDOWS, EVALUATION_WINDOWS, prepare_market_context
from backtester.evaluator import PortfolioConfig, StrategyEvaluator
from backtester.preview import floor_boundary, interval_to_seconds
from backtester.run_squeeze_v8_eval import _window_label
from backtester.models import PositionType
from live.squeeze_v8_strategy import SYMBOLS, SqueezeV8Strategy


def _window_set(name: str):
    if name == "development":
        return DEVELOPMENT_WINDOWS
    if name == "evaluation":
        return EVALUATION_WINDOWS
    return ALL_WINDOWS


def _select_windows(name: str, exact_window_name: str | None):
    windows = _window_set(name)
    if exact_window_name is None:
        return windows
    filtered = [window for window in windows if window.name == exact_window_name]
    if not filtered:
        raise ValueError(f"window {exact_window_name!r} was not found inside pack {name!r}")
    return filtered


def _build_strategy(
    analysis_interval: str,
    sizing_mode: str,
    poll_interval: str | None,
) -> tuple[SqueezeV8Strategy, str]:
    effective_sizing = sizing_mode
    if analysis_interval != "1h" and effective_sizing == "ridge_v1":
        print(
            "ridge_v1 is calibrated only for 1h candles; falling back to baseline sizing "
            f"for analysis_interval={analysis_interval}",
        )
        effective_sizing = "baseline"
    strategy = SqueezeV8Strategy(
        analysis_interval=analysis_interval,
        poll_interval=poll_interval or analysis_interval,
        sizing_mode=effective_sizing,
    )
    return strategy, effective_sizing


def _top_symbols_by_trade_count(report, limit: int) -> list[str]:
    counts = Counter()
    for wr in report.window_results:
        for trade in wr.backtest.trades:
            counts[trade.signal.ticker] += 1
    return [symbol for symbol, _ in counts.most_common(limit)]


def _collect_trades_by_symbol(report) -> dict[str, list]:
    by_symbol: dict[str, list] = defaultdict(list)
    for wr in report.window_results:
        for trade in wr.backtest.trades:
            by_symbol[trade.signal.ticker].append(trade)
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda trade: trade.entry_time)
    return dict(by_symbol)


def _collect_signals_by_symbol(report) -> dict[str, list]:
    by_symbol: dict[str, list] = defaultdict(list)
    for wr in report.window_results:
        for trade in wr.backtest.trades:
            by_symbol[trade.signal.ticker].append(trade.signal.signal_date)
    for symbol in by_symbol:
        by_symbol[symbol].sort()
    return dict(by_symbol)


def _plot_symbol_charts(report, strategy: SqueezeV8Strategy, windows, output_dir: Path, *, top_symbols: int) -> None:
    symbols = _top_symbols_by_trade_count(report, top_symbols)
    if not symbols:
        return

    start = min(window.start for window in windows)
    end = max(window.end for window in windows)
    ctx = prepare_market_context(
        symbols,
        start,
        end,
        request=strategy.market_data_request(),
        warmup_bars=strategy.warmup_bars,
    )
    trades_by_symbol = _collect_trades_by_symbol(report)

    plot_dir = output_dir / "symbol_charts"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        frame = ctx.for_symbol(symbol).frame
        if frame.empty or "close" not in frame.columns:
            continue
        plot_frame = frame[(frame["close_time"] >= start) & (frame["close_time"] <= end)].copy()
        if plot_frame.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(plot_frame["close_time"], plot_frame["close"], color="#264653", linewidth=1.2, label="Close")

        for trade in trades_by_symbol.get(symbol, []):
            color = "#2a9d8f" if trade.signal.position_type is PositionType.LONG else "#e76f51"
            marker = "^" if trade.signal.position_type is PositionType.LONG else "v"
            ax.scatter(trade.entry_time, trade.entry_price, color=color, marker=marker, s=60, zorder=4)
            ax.scatter(trade.exit_time, trade.exit_price, color=color, marker="x", s=40, zorder=4)
            ax.plot(
                [trade.entry_time, trade.exit_time],
                [trade.entry_price, trade.exit_price],
                color=color,
                linewidth=0.9,
                alpha=0.65,
            )

        ax.set_title(f"{symbol} | {_window_label('all' if len(windows) == len(ALL_WINDOWS) else 'evaluation' if windows == EVALUATION_WINDOWS else 'development')}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
        fig.tight_layout()
        safe_name = symbol.replace("/", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=160)
        plt.close(fig)


def _fuzzy_overlap_score(times_a, times_b, tolerance_seconds: int) -> float:
    if not times_a or not times_b:
        return 0.0
    i = 0
    j = 0
    matches = 0
    while i < len(times_a) and j < len(times_b):
        delta = (times_a[i] - times_b[j]).total_seconds()
        if abs(delta) <= tolerance_seconds:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matches / max(len(times_a), len(times_b))


def _plot_overlap_heatmap(report, analysis_interval: str, output_dir: Path) -> None:
    signals_by_symbol = _collect_signals_by_symbol(report)
    symbols = sorted(signals_by_symbol)
    if not symbols:
        return

    tolerance_seconds = interval_to_seconds(analysis_interval)
    matrix = np.zeros((len(symbols), len(symbols)), dtype=float)
    for i, sym_a in enumerate(symbols):
        for j, sym_b in enumerate(symbols):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = _fuzzy_overlap_score(
                    signals_by_symbol[sym_a],
                    signals_by_symbol[sym_b],
                    tolerance_seconds=tolerance_seconds,
                )

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(symbols)))
    ax.set_yticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=90)
    ax.set_yticklabels(symbols)
    ax.set_title(f"Signal Overlap Heatmap | tolerance = {analysis_interval}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "signal_overlap_heatmap.png", dpi=180)
    plt.close(fig)


def _write_overlap_summary(report, analysis_interval: str, output_dir: Path) -> None:
    signals_by_symbol = _collect_signals_by_symbol(report)
    tolerance_seconds = interval_to_seconds(analysis_interval)
    bucket_counts: Counter = Counter()
    for times in signals_by_symbol.values():
        for signal_time in times:
            bucket = floor_boundary(signal_time, analysis_interval)
            bucket_counts[bucket.isoformat()] += 1

    total_signals = sum(bucket_counts.values())
    active_buckets = len(bucket_counts)
    avg_cluster = (total_signals / active_buckets) if active_buckets else 0.0
    share_clustered_2 = (
        sum(count for count in bucket_counts.values() if count >= 2) / total_signals
        if total_signals else 0.0
    )
    share_clustered_3 = (
        sum(count for count in bucket_counts.values() if count >= 3) / total_signals
        if total_signals else 0.0
    )

    pair_scores: list[tuple[str, str, float]] = []
    symbols = sorted(signals_by_symbol)
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            pair_scores.append(
                (
                    sym_a,
                    sym_b,
                    _fuzzy_overlap_score(
                        signals_by_symbol[sym_a],
                        signals_by_symbol[sym_b],
                        tolerance_seconds=tolerance_seconds,
                    ),
                )
            )
    pair_scores.sort(key=lambda item: item[2], reverse=True)

    lines = [
        f"analysis_interval={analysis_interval}",
        f"total_signals={total_signals}",
        f"active_signal_timestamps={active_buckets}",
        f"avg_signals_per_timestamp={avg_cluster:.3f}",
        f"share_of_signals_in_clusters_ge_2={share_clustered_2:.3f}",
        f"share_of_signals_in_clusters_ge_3={share_clustered_3:.3f}",
        "",
        "top_overlap_pairs:",
    ]
    for sym_a, sym_b, score in pair_scores[:15]:
        lines.append(f"{sym_a} <-> {sym_b}: {score:.3f}")

    (output_dir / "signal_overlap_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Squeeze V8 backtests")
    parser.add_argument("--windows", choices=("development", "evaluation", "all"), default="evaluation")
    parser.add_argument("--window-name", type=str, default=None)
    parser.add_argument("--analysis-interval", choices=("1h", "30m", "15m", "5m", "1m"), default="1h")
    parser.add_argument("--poll-interval", choices=("1h", "30m", "15m", "5m", "1m"), default=None)
    parser.add_argument("--sizing-mode", choices=("ridge_v1", "baseline"), default="ridge_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--top-symbols", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="artifacts/squeeze_v8_visuals")
    args = parser.parse_args()

    windows = _select_windows(args.windows, args.window_name)
    strategy, sizing_mode = _build_strategy(
        args.analysis_interval,
        args.sizing_mode,
        args.poll_interval,
    )
    report = StrategyEvaluator(
        SYMBOLS,
        config=PortfolioConfig(approximate=not args.exact, seed=args.seed),
    ).evaluate(strategy, windows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.save(output_dir / "report")
    _plot_symbol_charts(report, strategy, windows, output_dir, top_symbols=args.top_symbols)
    _plot_overlap_heatmap(report, args.analysis_interval, output_dir)
    _write_overlap_summary(report, args.analysis_interval, output_dir)

    print(f"Saved visuals to {output_dir}")
    print(f"Analysis interval: {args.analysis_interval}")
    print(f"Poll interval: {strategy.effective_poll_interval}")
    print(f"Sizing mode: {sizing_mode}")
    print(f"Windows: {len(windows)}")
    print("Artifacts:")
    print(f"- {output_dir / 'report'}")
    print(f"- {output_dir / 'symbol_charts'}")
    print(f"- {output_dir / 'signal_overlap_heatmap.png'}")
    print(f"- {output_dir / 'signal_overlap_summary.txt'}")


if __name__ == "__main__":
    main()
