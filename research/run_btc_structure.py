from __future__ import annotations

import argparse
from pathlib import Path

from backtester.models import MarketType

from .btc_structure import BtcStructureConfig, run_btc_structure_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causal BTC market-structure pipeline and save artifacts.",
    )
    parser.add_argument("--ticker", default="BTC/USDT", help="Symbol to analyze. Default: BTC/USDT")
    parser.add_argument(
        "--market-type",
        choices=("futures", "spot"),
        default="futures",
        help="Binance market type for data fetch. Default: futures",
    )
    parser.add_argument("--interval", default="1d", help="Analysis interval. Default: 1d")
    parser.add_argument("--years", type=int, default=5, help="History length in years. Default: 5")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional artifact directory. Default: artifacts/btc_structure_<ticker>_<interval>_<years>y",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    market_type = MarketType.FUTURES if args.market_type == "futures" else MarketType.SPOT
    config = BtcStructureConfig.for_interval(
        args.interval,
        ticker=args.ticker,
        market_type=market_type,
        years=args.years,
    )
    sanitized_ticker = args.ticker.replace("/", "").lower()
    output_dir = args.output_dir or (
        Path("artifacts") / f"btc_structure_{sanitized_ticker}_{args.interval}_{args.years}y"
    )
    artifacts = run_btc_structure_pipeline(config, output_dir=output_dir)
    summary = artifacts.summary
    print(f"Ticker: {summary['ticker']} ({summary['market_type']})")
    print(f"Interval: {summary['interval']} | Bars: {summary['bars']}")
    print(f"Range: {summary['start']} -> {summary['end']}")
    print(f"Confirmed highs/lows: {summary['confirmed_highs']} / {summary['confirmed_lows']}")
    print(
        "Average bars to confirm: "
        f"high={summary['avg_bars_to_confirm_high']} "
        f"low={summary['avg_bars_to_confirm_low']}"
    )
    print(f"High labels: {summary['high_label_counts']}")
    print(f"Low labels: {summary['low_label_counts']}")
    print(f"Structure breaks: {summary['structure_break_counts']}")
    print(f"Latest bias after close: {summary['last_market_bias_after_close']}")
    print(f"Artifacts: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
