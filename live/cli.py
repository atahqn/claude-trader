from __future__ import annotations

import argparse
import os

from .models import LiveConfig


def add_live_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit futures testnet endpoint")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier (default: 1)")
    parser.add_argument(
        "--size",
        type=float,
        default=None,
        help="Position size in USDT (overrides config)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Max concurrent positions (overrides config)",
    )


def load_live_config_from_args(args: argparse.Namespace) -> LiveConfig:
    if getattr(args, "testnet", False):
        os.environ["BYBIT_TESTNET"] = "true"
    config = LiveConfig.load(getattr(args, "config", None))
    return config.with_overrides(
        use_testnet=bool(getattr(args, "testnet", False)),
        position_size_usdt=getattr(args, "size", None),
        max_concurrent_positions=getattr(args, "max_positions", None),
    )
