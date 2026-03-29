#!/usr/bin/env python3
"""Run the current live trading strategy.

Usage:
    python -m live.run [--config PATH] [--testnet] [--leverage N] [--size USDT]

Current strategy: Squeeze V8.1 (V8 signals + 72h max holding time)
"""

import argparse
import sys

from live.cli import add_live_runtime_args, load_live_config_from_args
from live.engine import LiveEngine
from live.squeeze_v8_strategy import SqueezeV8Strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Trader — Squeeze V8.1")
    add_live_runtime_args(parser)
    args = parser.parse_args()

    config = load_live_config_from_args(args)

    strategy = SqueezeV8Strategy(leverage=args.leverage)
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting Squeeze V8.1 Strategy\n"
        f"  Strategy: Squeeze SHORT + LONG\n"
        f"  SHORT:    TP/SL=3.0/1.5% (all regimes, mom<0, RSI>=25, ATR<=1.5)\n"
        f"  LONG:     TP/SL=4.0/2.0% (bull only: ret_72h>=6%, mom>0, RSI<=70)\n"
        f"  Shared:   min_squeeze=7 bars, cooldown=12h, max_hold={strategy.max_holding_hours}h\n"
        f"  Leverage: {args.leverage}x\n"
        f"  Size:     {config.position_size_usdt} USDT\n"
        f"  Max pos:  {config.max_concurrent_positions}\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
