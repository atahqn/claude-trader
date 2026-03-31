#!/usr/bin/env python3
"""Convenience wrapper for a guaranteed testnet entry with TP/SL."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced Bybit test trade with TP/SL")
    parser.add_argument("--testnet", action="store_true", help="Use testnet credentials")
    parser.add_argument("--symbol", type=str, default="DOGE/USDT", help="Trading symbol")
    parser.add_argument("--size-usdt", type=float, default=5.0, help="Position size in USDT")
    parser.add_argument("--tp-pct", type=float, default=10.0, help="TP percentage")
    parser.add_argument("--sl-pct", type=float, default=10.0, help="SL percentage")
    parser.add_argument("--observe-seconds", type=float, default=8.0, help="How long to keep the position open")
    parser.add_argument("--config", type=str, default=None, help="Optional config path")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "live.test_bybit_tp_sl",
        "--symbol",
        args.symbol,
        "--size-usdt",
        str(args.size_usdt),
        "--tp-pct",
        str(args.tp_pct),
        "--sl-pct",
        str(args.sl_pct),
        "--observe-seconds",
        str(args.observe_seconds),
        "--execute",
    ]
    if args.testnet:
        command.append("--testnet")
    if args.config:
        command.extend(["--config", args.config])

    env = dict(os.environ)
    if args.testnet:
        env["BYBIT_TESTNET"] = "true"

    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
