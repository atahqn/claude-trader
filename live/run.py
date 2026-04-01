#!/usr/bin/env python3
"""Run live trading with multiple strategies.

Usage:
    python -m live.run [--config PATH] [--testnet] [--leverage N] [--size USDT]

Strategies:
  - Squeeze V8.3: squeeze SHORT + pullback LONG (7 altcoins)
  - Breadth Long V21: cross-asset dip-buy LONG (11 symbols)
"""

import argparse
import sys

from live.breadth_long_strategy import BreadthLongStrategy
from live.cli import add_live_runtime_args, load_live_config_from_args
from live.engine import LiveEngine
from live.models import GeneratorBudget
from live.squeeze_v8_strategy import SqueezeV8Strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Trader — Multi-Strategy")
    add_live_runtime_args(parser)
    args = parser.parse_args()

    config = load_live_config_from_args(args)

    squeeze = SqueezeV8Strategy(leverage=args.leverage, sizing_mode="ridge_v1")
    breadth = BreadthLongStrategy(leverage=args.leverage)

    size = config.position_size_usdt
    max_pos = config.max_concurrent_positions
    budget = GeneratorBudget(position_size_usdt=size, max_positions=max_pos)

    engine = LiveEngine(
        generators=[
            (squeeze, budget),
            (breadth, budget),
        ],
        config=config,
    )

    print(
        f"Starting Multi-Strategy Live Trader\n"
        f"  Squeeze V8.3:  SHORT + LONG | 7 symbols\n"
        f"  Breadth V21:   LONG only    | 11 symbols\n"
        f"  Leverage: {args.leverage}x\n"
        f"  Size:     {size} USDT\n"
        f"  Max positions: {max_pos} (global ceiling)\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
