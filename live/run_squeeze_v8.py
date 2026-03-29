#!/usr/bin/env python3
"""Run the V8 Squeeze Strategy live (SHORT + LONG).

Usage:
    python -m live.run_squeeze_v8 [--config PATH] [--testnet] [--leverage N] [--size USDT]

V8 changes from V7:
  SHORT: TP 2.0→3.0, SL 1.0→1.5, RSI floor 30→25, ATR gate 1.3→1.5
  LONG:  unchanged (TP 4.0, SL 2.0, regime >= 6%)
"""

import argparse
import sys

sys.path.insert(0, "/home/caner/claude_trader")

from live.cli import add_live_runtime_args, load_live_config_from_args
from live.engine import LiveEngine
from live.squeeze_v8_strategy import SqueezeV8Strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Squeeze V8 - Live Trader")
    add_live_runtime_args(parser)

    args = parser.parse_args()

    config = load_live_config_from_args(args)

    strategy = SqueezeV8Strategy(leverage=args.leverage)
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting Squeeze V8 Strategy\n"
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
