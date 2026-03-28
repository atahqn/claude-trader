#!/usr/bin/env python3
"""Run the current live trading strategy.

Usage:
    python -m live.run [--config PATH] [--testnet] [--leverage N] [--size USDT]

Current strategy: Squeeze V8.1 (V8 signals + 72h max holding time)
"""

import argparse
import sys

sys.path.insert(0, "/home/caner/claude_trader")

from live.engine import LiveEngine
from live.models import LiveConfig
from live.squeeze_v8_strategy import SqueezeV8Strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Trader — Squeeze V8.1")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier (default: 1)")
    parser.add_argument("--size", type=float, default=None, help="Position size in USDT (overrides config)")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent positions (overrides config)")
    args = parser.parse_args()

    config = LiveConfig.load(args.config)

    overrides: dict = {}
    if args.testnet:
        overrides["testnet"] = True
    if args.size is not None:
        overrides["position_size_usdt"] = args.size
    if args.max_positions is not None:
        overrides["max_concurrent_positions"] = args.max_positions

    if overrides:
        config = LiveConfig(
            api_key=config.api_key,
            api_secret=config.api_secret,
            base_url=config.base_url,
            position_size_usdt=overrides.get("position_size_usdt", config.position_size_usdt),
            max_concurrent_positions=overrides.get("max_concurrent_positions", config.max_concurrent_positions),
            order_check_interval_seconds=config.order_check_interval_seconds,
            testnet=overrides.get("testnet", config.testnet),
        )

    strategy = SqueezeV8Strategy(leverage=args.leverage)
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting Squeeze V8.1 Strategy\n"
        f"  Strategy: Squeeze SHORT + LONG\n"
        f"  SHORT:    TP/SL=3.0/1.5% (all regimes, mom<0, RSI>=25, ATR<=1.5)\n"
        f"  LONG:     TP/SL=4.0/2.0% (bull only: ret_72h>=6%, mom>0, RSI<=70)\n"
        f"  Shared:   min_squeeze=7 bars, cooldown=12h, max_hold=72h\n"
        f"  Leverage: {args.leverage}x\n"
        f"  Size:     {config.position_size_usdt} USDT\n"
        f"  Max pos:  {config.max_concurrent_positions}\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
