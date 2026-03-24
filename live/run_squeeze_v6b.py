#!/usr/bin/env python3
"""Run the V6B Squeeze Strategy live (TRIP_R6T30).

Usage:
    python -m live.run_squeeze_v6b [--config PATH] [--testnet] [--leverage N] [--size USDT]

V6B keeps V6 squeeze SHORT + LONG and adds the DIP_TR long signal.
"""

import argparse
import sys

sys.path.insert(0, "/home/caner/claude-trader")

from live.engine import LiveEngine
from live.models import LiveConfig
from live.squeeze_v6b_strategy import SqueezeV6BStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Squeeze V6B - Live Trader")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier (default: 1)")
    parser.add_argument("--size", type=float, default=None, help="Position size in USDT (overrides config)")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent positions (overrides config)")
    parser.add_argument("--max-holding-hours", type=int, default=None, help="Max holding time per trade (overrides config)")
    args = parser.parse_args()

    config = LiveConfig.load(args.config)

    overrides: dict = {}
    if args.testnet:
        overrides["testnet"] = True
    if args.size is not None:
        overrides["position_size_usdt"] = args.size
    if args.max_positions is not None:
        overrides["max_concurrent_positions"] = args.max_positions
    if args.max_holding_hours is not None:
        overrides["max_holding_hours"] = args.max_holding_hours

    if overrides:
        config = LiveConfig(
            api_key=config.api_key,
            api_secret=config.api_secret,
            base_url=config.base_url,
            position_size_usdt=overrides.get("position_size_usdt", config.position_size_usdt),
            max_concurrent_positions=overrides.get("max_concurrent_positions", config.max_concurrent_positions),
            max_holding_hours=overrides.get("max_holding_hours", config.max_holding_hours),
            order_check_interval_seconds=config.order_check_interval_seconds,
            testnet=overrides.get("testnet", config.testnet),
        )

    strategy = SqueezeV6BStrategy(leverage=args.leverage)
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting Squeeze V6B Strategy\n"
        f"  Strategy: Squeeze SHORT + LONG + DIP_TR\n"
        f"  SHORT:    TP/SL=2.0/1.0% (all regimes, mom<0, RSI>=30, ATR<=1.3)\n"
        f"  LONG:     TP/SL=3.0/1.5% (bull only: ret_72h>=6%, mom>0, RSI<=70, ATR<=1.3)\n"
        f"  DIP_TR:   TP/SL=2.0/1.0% (ret_24h<=-3%, body>=0.2, vol>=1.2, 25<=RSI<=50, ret_72h>=0, ATR<=1.5)\n"
        f"  Shared:   squeeze min=7 bars, SHORT/LONG cooldown=12h, DIP cooldown=24h\n"
        f"  Leverage: {args.leverage}x\n"
        f"  Size:     {config.position_size_usdt} USDT\n"
        f"  Max pos:  {config.max_concurrent_positions}\n"
        f"  Hold max: {config.max_holding_hours}h\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
