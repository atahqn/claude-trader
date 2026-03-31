#!/usr/bin/env python3
"""Place and close a tiny Bybit test order outside the strategy runtime."""

from __future__ import annotations

import argparse
import os
import time
from decimal import ROUND_DOWN, ROUND_UP, Decimal

from .auth_client import BybitFuturesClient
from .models import LiveConfig, OrderSide, OrderStatus


def _find_symbol_info(client: BybitFuturesClient, symbol: str) -> dict:
    api_symbol = symbol.replace("/", "")
    info = client.get_exchange_info()
    for row in info.get("symbols", []):
        if row["symbol"] == api_symbol:
            return row
    raise ValueError(f"Symbol not found: {symbol}")


def _round_qty(symbol_info: dict, quantity: float) -> float:
    return _round_qty_with_mode(symbol_info, quantity, ROUND_DOWN)


def _round_qty_up(symbol_info: dict, quantity: float) -> float:
    return _round_qty_with_mode(symbol_info, quantity, ROUND_UP)


def _round_qty_with_mode(symbol_info: dict, quantity: float, rounding: str) -> float:
    lot_filter = symbol_info.get("lotSizeFilter", {})
    step = Decimal(str(lot_filter.get("qtyStep", "0")))
    if step <= 0:
        return quantity
    rounded = (Decimal(str(quantity)) / step).to_integral_value(rounding=rounding) * step
    return float(rounded)


def _minimum_qty(symbol_info: dict) -> float:
    lot_filter = symbol_info.get("lotSizeFilter", {})
    return float(lot_filter.get("minOrderQty", 0) or 0)


def _minimum_notional(symbol_info: dict) -> float:
    lot_filter = symbol_info.get("lotSizeFilter", {})
    return float(lot_filter.get("minNotionalValue", 0) or 0)


def _compute_qty(symbol_info: dict, mark_price: float, size_usdt: float) -> float:
    quantity = size_usdt / mark_price
    quantity = max(quantity, _minimum_qty(symbol_info))
    min_notional = _minimum_notional(symbol_info)
    if min_notional > 0 and quantity * mark_price < min_notional:
        quantity = min_notional / mark_price
    quantity = _round_qty_up(symbol_info, quantity)
    if min_notional > 0 and quantity * mark_price < min_notional:
        quantity = _round_qty_up(symbol_info, quantity + _minimum_qty(symbol_info))
    if quantity <= 0:
        raise ValueError("Rounded quantity is zero")
    return quantity


def _wait_for_fill(
    client: BybitFuturesClient,
    symbol: str,
    order_id: str,
    *,
    timeout_seconds: float = 10.0,
) -> tuple[float, float]:
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        order = client.get_order(symbol, order_id)
        last_status = order.status
        if order.status is OrderStatus.FILLED:
            fill_price = order.avg_fill_price if order.avg_fill_price > 0 else order.price
            fill_qty = order.filled_qty if order.filled_qty > 0 else order.quantity
            return fill_price, fill_qty
        time.sleep(0.5)
    raise RuntimeError(f"Order {order_id} did not fill in time (last status: {last_status})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Bybit order smoke test")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use testnet credentials")
    parser.add_argument("--symbol", type=str, default="DOGE/USDT", help="Trading symbol")
    parser.add_argument("--size-usdt", type=float, default=5.0, help="Target notional in USDT")
    parser.add_argument("--side", choices=("buy", "sell"), default="buy", help="Entry side")
    parser.add_argument("--close-after-seconds", type=float, default=2.0, help="How long to wait before closing")
    parser.add_argument("--execute", action="store_true", help="Actually place the order")
    args = parser.parse_args()

    if args.testnet:
        os.environ["BYBIT_TESTNET"] = "true"

    config = LiveConfig.load(args.config)
    client = BybitFuturesClient(config)
    symbol_info = _find_symbol_info(client, args.symbol)
    mark_price = client.get_mark_price(args.symbol)
    quantity = _compute_qty(symbol_info, mark_price, args.size_usdt)
    entry_side = OrderSide.BUY if args.side == "buy" else OrderSide.SELL
    close_side = OrderSide.SELL if entry_side is OrderSide.BUY else OrderSide.BUY

    print(f"Base URL: {config.base_url}")
    print(f"Symbol: {args.symbol}")
    print(f"Mark price: {mark_price}")
    print(f"Requested size: {args.size_usdt:.4f} USDT")
    print(f"Rounded quantity: {quantity}")
    print(f"Entry side: {entry_side.value}")
    print(f"Close side: {close_side.value}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to place the order.")
        return

    client.set_leverage(args.symbol, 1)
    entry = client.place_market_order(args.symbol, entry_side, quantity, position_side="BOTH")
    print(f"Entry order placed: {entry.order_id}")

    entry_fill_price, entry_fill_qty = _wait_for_fill(client, args.symbol, entry.order_id)
    print(f"Entry filled: qty={entry_fill_qty} price={entry_fill_price}")

    time.sleep(max(args.close_after_seconds, 0.0))

    close_order = client.place_market_order(args.symbol, close_side, entry_fill_qty, position_side="BOTH")
    print(f"Close order placed: {close_order.order_id}")

    close_fill_price, close_fill_qty = _wait_for_fill(client, args.symbol, close_order.order_id)
    print(f"Close filled: qty={close_fill_qty} price={close_fill_price}")
    print("Tiny order test completed.")


if __name__ == "__main__":
    main()
