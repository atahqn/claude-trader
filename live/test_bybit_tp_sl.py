#!/usr/bin/env python3
"""Open a tiny testnet position, attach TP/SL, observe it, then clean up."""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime

from backtester.models import PositionType, Signal
from backtester.resolver import compute_tp_sl_prices

from .executor import OrderExecutor
from .models import LiveConfig, OrderStatus, PositionStatus
from .auth_client import BybitFuturesClient


def _print_position_rows(rows: list[dict]) -> None:
    if not rows:
        print("Exchange position rows: 0")
        return
    print(f"Exchange position rows: {len(rows)}")
    for row in rows:
        print(
            "  "
            f"symbol={row.get('symbol')} side={row.get('side')} "
            f"size={row.get('size')} avgPrice={row.get('avgPrice')} "
            f"positionIdx={row.get('positionIdx')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Bybit TP/SL smoke test")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--testnet", action="store_true", help="Use testnet credentials")
    parser.add_argument("--symbol", type=str, default="DOGE/USDT", help="Trading symbol")
    parser.add_argument("--size-usdt", type=float, default=5.0, help="Position size in USDT")
    parser.add_argument("--tp-pct", type=float, default=10.0, help="TP percentage")
    parser.add_argument("--sl-pct", type=float, default=10.0, help="SL percentage")
    parser.add_argument("--observe-seconds", type=float, default=10.0, help="How long to keep position open before cleanup")
    parser.add_argument("--execute", action="store_true", help="Actually place the orders")
    args = parser.parse_args()

    if args.testnet:
        os.environ["BYBIT_TESTNET"] = "true"
    os.environ["BYBIT_POSITION_SIZE"] = str(args.size_usdt)

    config = LiveConfig.load(args.config)
    client = BybitFuturesClient(config)
    executor = OrderExecutor(client, config)

    signal = Signal(
        signal_date=datetime.now(UTC),
        position_type=PositionType.LONG,
        ticker=args.symbol,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        leverage=1.0,
        entry_price=None,
        entry_delay_seconds=0,
        fill_timeout_seconds=30,
        max_holding_hours=1,
        metadata={"smoke_test": "tp_sl"},
    )

    print(f"Base URL: {config.base_url}")
    print(f"Symbol: {args.symbol}")
    print(f"Size: {args.size_usdt} USDT")
    print(f"TP/SL: {args.tp_pct}% / {args.sl_pct}%")

    if not args.execute:
        print("Dry run only. Re-run with --execute to place the orders.")
        return

    position = executor.execute_signal(signal)
    print(f"Entry order placed: {position.entry_order.order_id if position.entry_order else 'unknown'}")

    if position.entry_order is None:
        raise RuntimeError("Entry order missing")

    entry = client.get_order(args.symbol, position.entry_order.order_id)
    if entry.status is not OrderStatus.FILLED:
        raise RuntimeError(f"Entry not filled immediately: {entry.status.value}")

    position.entry_order = entry
    position.status = PositionStatus.OPEN
    position.fill_price = entry.avg_fill_price if entry.avg_fill_price > 0 else entry.price
    position.quantity = entry.filled_qty if entry.filled_qty > 0 else position.quantity
    position.opened_at = entry.updated_at or entry.created_at or datetime.now(UTC)

    tp_price, sl_price = compute_tp_sl_prices(
        position.fill_price,
        signal.position_type,
        tp_pct=signal.tp_pct,
        sl_pct=signal.sl_pct,
        taker_fee_rate=signal.taker_fee_rate,
    )
    print(f"Entry filled: qty={position.quantity} price={position.fill_price}")
    print(f"Expected TP/SL prices: tp={tp_price} sl={sl_price}")

    executor.place_tp_sl(position)
    if position.tp_order is None or position.sl_order is None:
        raise RuntimeError("TP/SL orders were not created")

    print(f"TP order id: {position.tp_order.order_id} stop={position.tp_order.stop_price}")
    print(f"SL order id: {position.sl_order.order_id} stop={position.sl_order.stop_price}")

    tp_status = client.get_order(args.symbol, position.tp_order.order_id, conditional=True)
    sl_status = client.get_order(args.symbol, position.sl_order.order_id, conditional=True)
    print(f"TP status after placement: {tp_status.status.value}")
    print(f"SL status after placement: {sl_status.status.value}")

    rows = client.get_position_info(args.symbol)
    _print_position_rows(rows)

    print(f"Observing for {args.observe_seconds:.1f}s...")
    time.sleep(max(args.observe_seconds, 0.0))

    tp_status = client.get_order(args.symbol, position.tp_order.order_id, conditional=True)
    sl_status = client.get_order(args.symbol, position.sl_order.order_id, conditional=True)
    print(f"TP status after wait: {tp_status.status.value}")
    print(f"SL status after wait: {sl_status.status.value}")
    rows = client.get_position_info(args.symbol)
    _print_position_rows(rows)

    if tp_status.status is OrderStatus.NEW:
        client.cancel_order(args.symbol, position.tp_order.order_id, conditional=True)
        print("TP order canceled")
    if sl_status.status is OrderStatus.NEW:
        client.cancel_order(args.symbol, position.sl_order.order_id, conditional=True)
        print("SL order canceled")

    exit_order = executor.close_position_market(position)
    print(
        f"Manual close filled: qty={exit_order.filled_qty or exit_order.quantity} "
        f"price={exit_order.avg_fill_price or exit_order.price}"
    )
    print("TP/SL smoke test completed.")


if __name__ == "__main__":
    main()
