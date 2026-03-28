#!/usr/bin/env python3
"""Run the V8 Squeeze Strategy live (SHORT + LONG).

Usage:
    python -m live.run_squeeze_v8 [--config PATH] [--testnet] [--leverage N] [--size USDT]

V8 changes from V7:
  SHORT: TP 2.0→3.0, SL 1.0→1.5, RSI floor 30→25, ATR gate 1.3→1.5
  LONG:  unchanged (TP 4.0, SL 2.0, regime >= 6%)
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/caner/claude_trader")

from live.engine import LiveEngine
from live.models import LiveConfig
from live.squeeze_v8_strategy import SqueezeV8Strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Squeeze V8 - Live Trader")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier (default: 1)")
    parser.add_argument("--poll-interval", type=str, default=None, help="Optional poll interval override (1h, 30m, 15m, 5m, 1m)")
    parser.add_argument("--size", type=float, default=None, help="Position size in USDT (overrides config)")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent positions (overrides config)")
    parser.add_argument("--max-holding-hours", type=int, default=None, help="Max holding time per trade (overrides config)")
    args = parser.parse_args()

    if args.testnet:
        os.environ["BYBIT_TESTNET"] = "true"

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

    strategy = SqueezeV8Strategy(
        analysis_interval=config.analysis_interval,
        poll_interval=args.poll_interval if args.poll_interval is not None else config.poll_interval,
        leverage=args.leverage if args.leverage != 1.0 else config.leverage,
        short_tp=config.short_tp,
        short_sl=config.short_sl,
        short_cooldown_h=config.short_cooldown_h,
        short_rsi_floor=config.short_rsi_floor,
        long_tp=config.long_tp,
        long_sl=config.long_sl,
        long_cooldown_h=config.long_cooldown_h,
        long_rsi_cap=config.long_rsi_cap,
        long_regime_min=config.long_regime_min,
        min_squeeze_bars=config.min_squeeze_bars,
        atr_ratio_max=config.atr_ratio_max,
    )
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting Squeeze V8 Strategy\n"
        f"  Strategy: Squeeze SHORT + LONG\n"
        f"  SHORT:    TP/SL=3.0/1.5% (all regimes, mom<0, RSI>=25, ATR<=1.5)\n"
        f"  LONG:     TP/SL=4.0/2.0% (bull only: ret_72h>=6%, mom>0, RSI<=70)\n"
        f"  Shared:   min_squeeze=7 bars, cooldown=12h\n"
        f"  Leverage: {strategy.leverage}x\n"
        f"  Poll:     {strategy.effective_poll_interval}\n"
        f"  Size:     {config.position_size_usdt} USDT\n"
        f"  Max pos:  {config.max_concurrent_positions}\n"
        f"  Hold max: {config.max_holding_hours}h\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
