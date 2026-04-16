from __future__ import annotations

import argparse
from pathlib import Path

from backtester.models import MarketType

from .btc_structure import BtcStructureConfig, run_btc_structure_pipeline
from .structure_feature_lab import run_structure_feature_lab, save_structure_feature_lab


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ranked structure features for BTC research experiments.",
    )
    parser.add_argument("--ticker", default="BTC/USDT")
    parser.add_argument("--market-type", choices=("futures", "spot"), default="futures")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
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
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("artifacts") / f"btc_structure_{sanitized_ticker}_{args.interval}_{args.years}y"
    )
    artifacts = run_btc_structure_pipeline(config, output_dir=output_dir)
    lab = run_structure_feature_lab(artifacts)
    save_structure_feature_lab(lab, output_dir)
    print(f"Ranked highs: {len(lab.ranked_highs)}")
    print(f"Ranked lows: {len(lab.ranked_lows)}")
    print(f"Ranked breaks: {len(lab.ranked_breaks)}")
    print(f"Feature matrix shape: {lab.feature_matrix.shape}")
    print(f"Artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
