#!/usr/bin/env python3
"""Evaluate a live SignalGenerator on the canonical weekly window packs.

Example:
    python run_strategy_eval.py \
        --strategy live/squeeze_v8_strategy.py:SqueezeV8Strategy \
        --windows eval \
        --exact
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester import PortfolioConfig, StrategyEvaluator
from backtester.eval_windows import (
    ALL_WINDOWS,
    BULL_DEVELOPMENT_WINDOWS,
    DEVELOPMENT_WINDOWS,
    EVALUATION_WINDOWS,
    HOLDOUT_WINDOWS,
    LEGACY_DEVELOPMENT_WINDOWS,
    OOS2_WINDOWS,
    OOS3_WINDOWS,
    OOS4_WINDOWS,
    OOS5_WINDOWS,
    PAIRED_DEVELOPMENT_WINDOWS,
    RANDOM_DEVELOPMENT_WINDOWS,
    RANDOM_EVALUATION_WINDOWS,
    STRESS_DEVELOPMENT_WINDOWS,
    EvalWindow,
)
from live.signal_generator import CompositeSignalGenerator, SignalGenerator


WINDOW_ALIASES: dict[str, list[EvalWindow]] = {
    "all": ALL_WINDOWS,
    "dev": DEVELOPMENT_WINDOWS,
    "development": DEVELOPMENT_WINDOWS,
    "eval": EVALUATION_WINDOWS,
    "evaluation": EVALUATION_WINDOWS,
    "holdout": HOLDOUT_WINDOWS,
    "oos2": OOS2_WINDOWS,
    "oos3": OOS3_WINDOWS,
    "oos4": OOS4_WINDOWS,
    "oos5": OOS5_WINDOWS,
    "legacy_development": LEGACY_DEVELOPMENT_WINDOWS,
    "development_stress": STRESS_DEVELOPMENT_WINDOWS,
    "development_bull": BULL_DEVELOPMENT_WINDOWS,
    "development_pairs": PAIRED_DEVELOPMENT_WINDOWS,
    "development_random": RANDOM_DEVELOPMENT_WINDOWS,
    "evaluation_random": RANDOM_EVALUATION_WINDOWS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a live SignalGenerator and save detailed results.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        nargs="+",
        help=(
            "One or more strategy specs as module[:attr] or path/to/file.py[:attr]. "
            "If attr is omitted, a unique SignalGenerator subclass defined in the file is used. "
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
        default="eval",
        help=(
            "Window pack: eval, dev, all, holdout, oos2, oos3, oos4, oos5, "
            "legacy_development, development_stress, development_bull, "
            "development_pairs, development_random, evaluation_random."
        ),
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol list. Defaults to module-level SYMBOLS.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--approximate",
        dest="approximate",
        action="store_true",
        help="Use approximate execution.",
    )
    mode.add_argument(
        "--exact",
        dest="approximate",
        action="store_false",
        help="Use exact execution. This is the default when neither mode flag is passed.",
    )
    parser.set_defaults(approximate=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional execution seed passed to the backtester.",
    )
    parser.add_argument(
        "--risk-free-rate-annual",
        type=float,
        default=None,
        help="Override the evaluator Sortino risk-free rate.",
    )
    parser.add_argument(
        "--cooldown-warmup-days",
        type=int,
        default=14,
        help="Period merge threshold used by StrategyEvaluator.",
    )
    parser.add_argument(
        "--entry-delay-seconds",
        type=int,
        default=None,
        help=(
            "Override the backtester default market-entry delay in seconds. "
            "Applies when signals do not set entry_delay_seconds explicitly."
        ),
    )
    parser.add_argument(
        "--data-max-workers",
        type=int,
        default=8,
        help="Max threads for parallel symbol data fetching (default: 8).",
    )
    parser.add_argument(
        "--backtest-max-workers",
        type=int,
        default=0,
        help="Max threads for parallel signal backtesting (0 = auto, caps at 16; 1 = sequential).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to outputs/strategy_eval/<strategy>_<windows>_<mode>_<timestamp>.",
    )
    parser.add_argument(
        "--equity-curve",
        action="store_true",
        default=False,
        help="Generate a granular equity curve plot from intermediate candle data.",
    )
    parser.add_argument(
        "--equity-interval",
        default="15m",
        help="Candle interval for equity curve granularity (default: 15m).",
    )
    parser.add_argument(
        "--position-size-usdt",
        type=float,
        default=10_000.0,
        help="Position size in USDT per trade for equity curve (default: 10000).",
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=10_000.0,
        help="Starting capital in USDT for equity curve (default: 10000).",
    )
    return parser.parse_args()


def resolve_windows(selector: str) -> tuple[str, list[EvalWindow]]:
    key = selector.strip().lower()
    if key in WINDOW_ALIASES:
        return key, WINDOW_ALIASES[key]

    matches = [w for w in ALL_WINDOWS if w.category.lower() == key]
    if matches:
        return key, matches

    supported = ", ".join(sorted(WINDOW_ALIASES))
    raise ValueError(f"Unknown window selector '{selector}'. Supported: {supported}")


def parse_strategy_kwargs(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --strategy-kwargs JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--strategy-kwargs must decode to a JSON object")
    return parsed


def _normalize_module_name(raw_spec: str) -> str:
    raw_spec = raw_spec.strip()
    candidate = Path(raw_spec)
    if candidate.suffix == ".py" or candidate.exists():
        path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
        path = path.resolve()
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"Strategy file '{path}' must live under the project root {PROJECT_ROOT}"
            ) from exc
        if rel.suffix != ".py":
            raise ValueError(f"Strategy path '{raw_spec}' is not a Python file")
        rel_no_suffix = rel.with_suffix("")
        if rel_no_suffix.name == "__init__":
            rel_no_suffix = rel_no_suffix.parent
        return ".".join(rel_no_suffix.parts)
    return raw_spec


def _infer_strategy_attr(module: Any) -> Any:
    local_subclasses = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, SignalGenerator)
        and obj is not SignalGenerator
        and obj.__module__ == module.__name__
    ]
    if len(local_subclasses) == 1:
        return local_subclasses[0]
    if not local_subclasses:
        raise ValueError(
            f"Module '{module.__name__}' does not define a SignalGenerator subclass. "
            "Pass --strategy module.py:ClassName explicitly."
        )
    names = ", ".join(cls.__name__ for cls in local_subclasses)
    raise ValueError(
        f"Module '{module.__name__}' defines multiple SignalGenerator subclasses: {names}. "
        "Pass --strategy module.py:ClassName explicitly."
    )


def load_strategy(strategy_spec: str, kwargs: dict[str, Any]) -> tuple[SignalGenerator, Any, str]:
    module_spec, _, attr_name = strategy_spec.partition(":")
    module_name = _normalize_module_name(module_spec)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name) if attr_name else _infer_strategy_attr(module)

    if isinstance(target, SignalGenerator):
        if kwargs:
            raise ValueError("Strategy spec resolved to an instance; --strategy-kwargs cannot be used")
        return target, module, type(target).__name__

    if inspect.isclass(target):
        if not issubclass(target, SignalGenerator):
            raise ValueError(f"{module_name}:{target.__name__} is not a SignalGenerator subclass")
        return target(**kwargs), module, target.__name__

    if callable(target):
        instance = target(**kwargs)
        if not isinstance(instance, SignalGenerator):
            raise ValueError(
                f"Factory '{module_name}:{getattr(target, '__name__', attr_name or '<callable>')}' "
                "did not return a SignalGenerator"
            )
        return instance, module, type(instance).__name__

    raise ValueError(f"Unsupported strategy target '{module_name}:{attr_name or '<inferred>'}'")


def resolve_symbols(raw_symbols: str, module: Any, strategy: SignalGenerator) -> list[str]:
    if raw_symbols.strip():
        return [s.strip() for s in raw_symbols.split(",") if s.strip()]

    for container in (module, strategy, type(strategy)):
        symbols = getattr(container, "SYMBOLS", None)
        if isinstance(symbols, list) and all(isinstance(sym, str) for sym in symbols):
            return list(symbols)

    raise ValueError(
        "Could not infer symbols from the strategy module. Pass --symbols BTC/USDT,ETH/USDT,... explicitly."
    )


def default_output_dir(
    strategy_name: str,
    window_label: str,
    approximate: bool,
) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", strategy_name).strip("_.-") or "strategy"
    mode = "approx" if approximate else "exact"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "outputs" / "strategy_eval" / f"{slug}_{window_label}_{mode}_{stamp}"


def _summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "category": summary.category,
        "windows": summary.windows,
        "total_pnl": round(summary.total_pnl, 4),
        "weekly_win_rate": round(summary.weekly_win_rate, 4),
        "profit_factor": _json_metric(summary.profit_factor),
        "sortino_ratio": _json_metric(summary.sortino_ratio),
        "max_drawdown_pct": round(summary.max_drawdown_pct, 4),
        "pnl_to_mdd": _json_metric(summary.pnl_to_mdd),
        "weekly_omega_ratio": _json_metric(summary.weekly_omega_ratio),
        "coverage_penalty": round(summary.coverage_penalty, 4),
        "preference_eligible": summary.preference_eligible,
        "preference_score": _json_metric(summary.preference_score),
        "total_trades": summary.total_trades,
        "resolved_trades": summary.resolved_trades,
        "active_weeks": summary.active_weeks,
    }


def _json_metric(value: float) -> float | str:
    if math.isfinite(value):
        return round(value, 4)
    return "inf" if value > 0 else "-inf"


def enrich_meta(
    meta_path: Path,
    *,
    strategy_spec: str,
    strategy_name: str,
    strategy_kwargs: dict[str, Any],
    module_name: str,
    window_selector: str,
    symbols: list[str],
    summary: Any,
) -> None:
    meta = json.loads(meta_path.read_text())
    meta["strategy"] = {
        "spec": strategy_spec,
        "module": module_name,
        "class_name": strategy_name,
        "kwargs": strategy_kwargs,
    }
    meta["window_selector"] = window_selector
    meta["run_symbols"] = symbols
    meta["selected_summary"] = _summary_to_dict(summary)
    meta_path.write_text(json.dumps(meta, indent=2))


def print_run_summary(
    *,
    strategy_spec: str,
    strategy_name: str,
    window_selector: str,
    approximate: bool,
    symbols: list[str],
    output_dir: Path,
    report_summary: Any,
    table: str,
    resolution_breakdown: tuple[int, int] | None = None,
) -> None:
    print(table)
    print()
    print(f"Strategy:   {strategy_name} ({strategy_spec})")
    print(f"Windows:    {window_selector} ({report_summary.windows} weeks)")
    print(f"Mode:       {'approximate' if approximate else 'exact'}")
    print(f"Symbols:    {len(symbols)}")
    print(f"Preference: {_display_metric(report_summary.preference_score)}")
    print(
        "Summary:    "
        f"PNL {report_summary.total_pnl:+.2f}% | "
        f"MDD {report_summary.max_drawdown_pct:.2f}% | "
        f"Omega {_display_metric(report_summary.weekly_omega_ratio)} | "
        f"PF {_display_metric(report_summary.profit_factor)} | "
        f"Sortino {_display_metric(report_summary.sortino_ratio)}"
    )
    if resolution_breakdown is not None:
        exact_resolved, fallback_resolved = resolution_breakdown
        print(f"Resolved:   {exact_resolved} exact | {fallback_resolved} fallback")
    print(f"Eligible:   {report_summary.preference_eligible}")
    print(f"Saved to:   {output_dir}")


def _display_metric(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.2f}"
    return "inf" if value > 0 else "-inf"


def _load_strategies(
    specs: list[str], kwargs: dict[str, Any],
) -> tuple[SignalGenerator, str, str]:
    """Load one or more strategy specs and return (generator, display_name, strategy_spec_str).

    For a single spec, returns the generator directly.
    For multiple specs, wraps them in a CompositeSignalGenerator.
    kwargs are only applied when a single strategy is loaded.
    """
    if len(specs) == 1:
        strategy, module, name = load_strategy(specs[0], kwargs)
        return strategy, name, specs[0]

    if kwargs and kwargs != {}:
        raise ValueError("--strategy-kwargs is not supported with multiple strategies")

    generators: list[SignalGenerator] = []
    names: list[str] = []
    for spec in specs:
        gen, _, name = load_strategy(spec, {})
        generators.append(gen)
        names.append(name)

    composite = CompositeSignalGenerator(generators)
    display_name = "+".join(names)
    spec_str = " ".join(specs)
    return composite, display_name, spec_str


def main() -> int:
    args = parse_args()
    try:
        strategy_kwargs = parse_strategy_kwargs(args.strategy_kwargs)
        window_label, windows = resolve_windows(args.windows)
        strategy, strategy_name, strategy_spec = _load_strategies(
            args.strategy, strategy_kwargs,
        )
        symbols = (
            [s.strip() for s in args.symbols.split(",") if s.strip()]
            if args.symbols.strip()
            else strategy.symbols
        )

        config = PortfolioConfig(
            approximate=args.approximate,
            seed=args.seed,
            risk_free_rate_annual=(
                args.risk_free_rate_annual
                if args.risk_free_rate_annual is not None
                else PortfolioConfig().risk_free_rate_annual
            ),
            entry_delay_seconds=(
                args.entry_delay_seconds
                if args.entry_delay_seconds is not None
                else PortfolioConfig().entry_delay_seconds
            ),
            data_max_workers=args.data_max_workers,
            backtest_max_workers=args.backtest_max_workers,
        )
        # Build a strategy factory for parallel signal generation.
        strategy_factory = None
        if len(args.strategy) == 1:
            spec = args.strategy[0]
            module_spec, _, attr_name = spec.partition(":")
            module_name = _normalize_module_name(module_spec)
            if attr_name:
                strategy_factory = (module_name, attr_name, strategy_kwargs)
        else:
            # Composite: list of (module, class, {}) tuples.
            parts = []
            for spec in args.strategy:
                module_spec, _, attr_name = spec.partition(":")
                if not attr_name:
                    parts = None
                    break
                parts.append((_normalize_module_name(module_spec), attr_name, {}))
            strategy_factory = parts

        evaluator = StrategyEvaluator(
            symbols=symbols,
            config=config,
            cooldown_warmup=timedelta(days=args.cooldown_warmup_days),
            strategy_factory=strategy_factory,
        )
        report = evaluator.evaluate(strategy, windows)
        summary = report.overall_summary()
        resolution_breakdown = (
            report.resolved_trade_breakdown()
            if not config.approximate
            else None
        )

        output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
            strategy_name,
            window_label,
            config.approximate,
        )
        saved_dir = report.save(output_dir)
        enrich_meta(
            saved_dir / "meta.json",
            strategy_spec=strategy_spec,
            strategy_name=strategy_name,
            strategy_kwargs=strategy_kwargs,
            module_name=strategy_name,
            window_selector=window_label,
            symbols=symbols,
            summary=summary,
        )
        print_run_summary(
            strategy_spec=strategy_spec,
            strategy_name=strategy_name,
            window_selector=window_label,
            approximate=config.approximate,
            symbols=symbols,
            output_dir=saved_dir,
            report_summary=summary,
            table=report.format_table(),
            resolution_breakdown=resolution_breakdown,
        )

        if args.equity_curve:
            from backtester.data import BinanceClient
            from backtester.equity_curve import (
                compute_granular_equity_curve,
                plot_equity_curve,
                save_equity_csv,
            )

            all_trades = [t for wr in report.window_results for t in wr.backtest.trades]
            client = BinanceClient()
            points = compute_granular_equity_curve(
                all_trades,
                client,
                interval=args.equity_interval,
                position_size_usdt=args.position_size_usdt,
                starting_capital=args.starting_capital,
                max_workers=args.data_max_workers,
            )
            if points:
                curve_title = f"{strategy_name} \u2014 Equity Curve ({args.equity_interval})"
                plot_path = plot_equity_curve(
                    points,
                    saved_dir / "equity_curve.html",
                    title=curve_title,
                    starting_capital=args.starting_capital,
                    interval=args.equity_interval,
                )
                csv_path = save_equity_csv(points, saved_dir / "equity_curve.csv")
                print(f"Equity curve: {plot_path}")
                print(f"Equity data:  {csv_path}")

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
