#!/usr/bin/env python3
"""Non-trading Bybit credential smoke test."""

from __future__ import annotations

import argparse
import os
import sys

from .auth_client import BybitFuturesClient
from .models import LiveConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Bybit credentials without placing orders")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use testnet credentials")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol used for mark-price check")
    args = parser.parse_args()

    if args.testnet:
        os.environ["BYBIT_TESTNET"] = "true"

    config = LiveConfig.load(args.config)
    client = BybitFuturesClient(config)

    print(f"Base URL: {config.base_url}")
    print(f"Testnet: {config.testnet}")
    print(f"Server time: {client.server_now().isoformat()}")

    balance = client.get_available_balance()
    print(f"Available balance: {balance:.8f} USDT")

    mark_price = client.get_mark_price(args.symbol)
    print(f"{args.symbol} mark price: {mark_price}")

    positions = client.get_position_info()
    print(f"Position query ok: {len(positions)} row(s)")

    exchange_info = client.get_exchange_info()
    print(f"Instruments query ok: {len(exchange_info.get('symbols', []))} symbol(s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Credential check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
