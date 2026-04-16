#!/usr/bin/env python3
"""Validate a SignalGenerator for look-ahead bias on selected windows.

Example:
    python run_strategy_validate.py \
        --strategy adaptive_regime_momentum/strategy.py:AdaptiveRegimeMomentum \
        --windows development
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester.calibration import validate_calibration_config
from backtester.eval_windows import EvalWindow
from backtester.pipeline import prepare_market_context
from backtester.preview import interval_to_timedelta
from backtester.validation import LookaheadViolation, validate_no_lookahead
from run_strategy_eval import _load_strategies, parse_strategy_kwargs, resolve_windows


@dataclass(frozen=True, slots=True)
class ValidationResult:
    window: EvalWindow
    violations: list[LookaheadViolation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a SignalGenerator for look-ahead bias.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        nargs="+",
        help=(
            "One or more strategy specs as module[:attr] or path/to/file.py[:attr]. "
            "When multiple specs are given, strategies are combined into a composite."
        ),
    )
    parser.add_argument(
        "--strategy-kwargs",
        default="{}",
        help="JSON dict passed to the strategy constructor/factory.",
    )
    parser.add_argument(
        "--windows",
        default="development",
        help="Window pack to validate on (default: development).",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Custom period start date (YYYY-MM-DD). Used with --end.",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Custom period end date (YYYY-MM-DD). Used with --start.",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol list. Defaults to strategy.symbols.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Signals to validate per window (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--data-max-workers",
        type=int,
        default=8,
        help="Max threads for parallel symbol data fetching (default: 8).",
    )
    return parser.parse_args()


def _resolve_validation_windows(args: argparse.Namespace) -> tuple[str, list[EvalWindow]]:
    if args.start and args.end:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
        if end_dt <= start_dt:
            raise ValueError("--end must be after --start")
        label = f"{args.start}_to_{args.end}"
        return label, [EvalWindow(label, start_dt, end_dt, "custom")]
    if args.start or args.end:
        raise ValueError("Both --start and --end are required for a custom period")
    return resolve_windows(args.windows)


def _compute_warmup_bars(strategy: object) -> tuple[tuple[str, ...], int]:
    request = strategy.market_data_request()
    indicators = strategy.indicator_request()
    warmup_bars = strategy.required_warmup_bars

    if indicators:
        from backtester.indicators import required_warmup as indicator_required_warmup

        warmup_bars = max(warmup_bars, indicator_required_warmup(indicators))

    if strategy.needs_calibration:
        validate_calibration_config(strategy)
        interval_td = interval_to_timedelta(request.ohlcv_interval)
        interval_hours = interval_td.total_seconds() / 3600
        warmup_bars = max(
            warmup_bars,
            int(strategy.calibration_lookback_hours / interval_hours),
        )

    return indicators, warmup_bars


def _print_summary(
    *,
    strategy_spec: str,
    strategy_name: str,
    window_selector: str,
    symbols: list[str],
    sample_size: int,
    results: list[ValidationResult],
) -> None:
    total_violations = sum(len(result.violations) for result in results)
    print(f"Strategy:   {strategy_name} ({strategy_spec})")
    print(f"Windows:    {window_selector} ({len(results)} windows)")
    print(f"Symbols:    {len(symbols)}")
    print(f"SampleSize: {sample_size} per window")
    print(f"Result:     {'PASS' if total_violations == 0 else 'FAIL'}")
    print()

    for result in results:
        window = result.window
        status = "PASS" if not result.violations else f"FAIL ({len(result.violations)} violations)"
        start = window.start.strftime("%Y-%m-%d")
        end = window.end.strftime("%Y-%m-%d")
        print(f"{window.category:16} {start} -> {end}  {status}")
        for violation in result.violations:
            print(f"  - {violation.detail}")


def main() -> int:
    args = parse_args()
    try:
        if args.sample_size < 1:
            raise ValueError("--sample-size must be at least 1")

        strategy_kwargs = parse_strategy_kwargs(args.strategy_kwargs)
        window_label, windows = _resolve_validation_windows(args)
        strategy, strategy_name, strategy_spec = _load_strategies(
            args.strategy, strategy_kwargs,
        )
        symbols = (
            [s.strip() for s in args.symbols.split(",") if s.strip()]
            if args.symbols.strip()
            else strategy.symbols
        )

        request = strategy.market_data_request()
        indicators, warmup_bars = _compute_warmup_bars(strategy)

        results: list[ValidationResult] = []
        for window in windows:
            ctx = prepare_market_context(
                symbols,
                window.start,
                window.end,
                request=request,
                warmup_bars=warmup_bars,
                indicators=indicators,
                max_workers=args.data_max_workers,
            )
            violations = validate_no_lookahead(
                strategy,
                ctx,
                symbols,
                window.start,
                window.end,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            results.append(ValidationResult(window=window, violations=violations))

        _print_summary(
            strategy_spec=strategy_spec,
            strategy_name=strategy_name,
            window_selector=window_label,
            symbols=symbols,
            sample_size=args.sample_size,
            results=results,
        )

        total_violations = sum(len(result.violations) for result in results)
        return 0 if total_violations == 0 else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
