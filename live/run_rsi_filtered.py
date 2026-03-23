#!/usr/bin/env python3
"""Run the V3 RSI-filtered asymmetric regime ensemble strategy live.

Usage:
    python -m live.run_rsi_filtered [--config PATH] [--testnet] [--leverage N] [--size USDT]

Configuration via --config, env vars, or ~/.claude_trader/live_config.json:
    BINANCE_API_KEY, BINANCE_API_SECRET (required)
    BINANCE_TESTNET=true            (use testnet)
    BINANCE_POSITION_SIZE=100       (USDT per trade)
    BINANCE_MAX_POSITIONS=3         (concurrent positions)
    BINANCE_MAX_HOLDING_HOURS=168   (max holding time per trade)
    BINANCE_ORDER_CHECK_INTERVAL=5  (seconds between engine checks)
"""

import argparse
import sys

sys.path.insert(0, "/home/caner/claude_trader")

from live.engine import LiveEngine
from live.models import LiveConfig
from live.rsi_filtered_strategy import RsiFilteredStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="RSI-Filtered Regime Ensemble V3 - Live Trader")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier (default: 1)")
    parser.add_argument("--size", type=float, default=None, help="Position size in USDT (overrides config)")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent positions (overrides config)")
    parser.add_argument("--max-holding-hours", type=int, default=None, help="Max holding time per trade (overrides config)")
    args = parser.parse_args()

    # Load base config from an explicit file or the default env/file lookup.
    config = LiveConfig.load(args.config)

    # Apply CLI overrides by rebuilding config
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

    strategy = RsiFilteredStrategy(leverage=args.leverage)
    engine = LiveEngine(generator=strategy, config=config)

    print(
        f"Starting RSI-Filtered Regime Ensemble V3\n"
        f"  Strategy: Squeeze SHORT-only (RSI>=30) + Vol Spike BULLISH-only (LONG RSI<=75)\n"
        f"  Regime:   72h return (squeeze<2%, vs>0%)\n"
        f"  Vol min:  1.8x\n"
        f"  Leverage: {args.leverage}x\n"
        f"  Size:     {config.position_size_usdt} USDT\n"
        f"  Max pos:  {config.max_concurrent_positions}\n"
        f"  Hold max: {config.max_holding_hours}h\n"
        f"  Checks:   {config.order_check_interval_seconds}s\n"
        f"  Testnet:  {config.testnet}\n",
        file=sys.stderr,
    )

    engine.start()


if __name__ == "__main__":
    main()
