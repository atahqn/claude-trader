#!/usr/bin/env python3
"""Run the current live trading strategy.

Usage:
    python -m live.run [--config PATH] [--testnet] [--leverage N] [--size USDT]

Current strategy: Squeeze V8.3 (squeeze SHORT + pullback LONG + ridge_v1 dynamic sizing)
"""

import argparse
import sys

from live.cli import add_live_runtime_args, load_live_config_from_args
from live.engine import LiveEngine
from live.squeeze_v8_strategy import SqueezeV8Strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Trader — Squeeze V8.3")
    add_live_runtime_args(parser)
    args = parser.parse_args()

    config = load_live_config_from_args(args)

    strategy = SqueezeV8Strategy(leverage=args.leverage, sizing_mode="ridge_v1")
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting Squeeze V8.3 Strategy\n"
        f"  Strategy: Squeeze SHORT (tiered TP/SL) + bull pullback reclaim LONG\n"
        f"  SHORT:    TP/SL=tiered by quality (A:3.25/1.5 B:3.0/1.5 C:3.0/1.1)\n"
        f"  LONG:     TP/SL=4.0/2.0% (10%<=ret_72h<=25%, mom>0, RSI<=75, ATR<=1.2)\n"
        f"  Shared:   cooldown=12h, max_hold={strategy.max_holding_hours}h, short priority on conflicts\n"
        f"  Sizing:   {strategy.sizing_mode}\n"
        f"  Leverage: {args.leverage}x\n"
        f"  Size:     {config.position_size_usdt} USDT\n"
        f"  Max pos:  {config.max_concurrent_positions}\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
