from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtester.models import PositionType, Signal
from backtester.resolver import compute_pnl

from .auth_client import BinanceFuturesClient
from .executor import OrderExecutor
from .models import (
    ExchangeOrder,
    LivePosition,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
)

_STATE_DIR = Path.home() / ".claude_trader"
_STATE_PATH = _STATE_DIR / "live_state.json"


class PositionTracker:
    """Manages position lifecycle: PENDING_ENTRY → OPEN → CLOSED / FAILED."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        executor: OrderExecutor,
    ) -> None:
        self._client = client
        self._executor = executor
        self._positions: list[LivePosition] = []
        self._dirty = False

    @property
    def positions(self) -> list[LivePosition]:
        return self._positions

    @property
    def open_count(self) -> int:
        return sum(
            1 for p in self._positions
            if p.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)
        )

    # -- Position management ---------------------------------------------------

    def add_position(self, position: LivePosition) -> None:
        self._positions.append(position)
        self._dirty = True

    def check_fills(self) -> None:
        """Poll exchange for order status updates on all active positions."""
        for pos in self._positions:
            if pos.status is PositionStatus.PENDING_ENTRY:
                self._check_entry_fill(pos)
            elif pos.status is PositionStatus.OPEN:
                self._check_exit_fills(pos)

    # -- Entry fill detection --------------------------------------------------

    def _check_entry_fill(self, pos: LivePosition) -> None:
        if pos.entry_order is None:
            pos.status = PositionStatus.FAILED
            self._dirty = True
            return

        updated = self._client.get_order(pos.signal.ticker, pos.entry_order.order_id)

        # Cancel stale limit orders that haven't filled within the timeout
        if (
            updated.status is OrderStatus.NEW
            and updated.created_at is not None
            and pos.signal.fill_timeout_seconds > 0
        ):
            age = (datetime.now(UTC) - updated.created_at).total_seconds()
            if age >= pos.signal.fill_timeout_seconds:
                try:
                    self._client.cancel_order(pos.signal.ticker, updated.order_id)
                except Exception:
                    pass
                pos.status = PositionStatus.FAILED
                pos.entry_order = updated
                self._dirty = True
                print(
                    f"[{pos.position_id}] Entry order timed out for {pos.signal.ticker} "
                    f"after {age:.0f}s — canceled.",
                    file=sys.stderr,
                )
                return

        if updated.status is OrderStatus.FILLED:
            pos.fill_price = updated.avg_fill_price if updated.avg_fill_price > 0 else updated.price
            pos.quantity = updated.filled_qty if updated.filled_qty > 0 else pos.quantity
            pos.entry_order = updated
            self._dirty = True
            # Place TP/SL orders on the exchange
            try:
                self._executor.place_tp_sl(pos)
                pos.status = PositionStatus.OPEN
                print(
                    f"[{pos.position_id}] Entry filled {pos.signal.ticker} "
                    f"@ {pos.fill_price}, TP/SL placed.",
                    file=sys.stderr,
                )
                try:
                    balance = self._client.get_available_balance()
                    print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)
                except Exception:
                    pass
            except Exception as exc:
                print(
                    f"[{pos.position_id}] Entry filled but TP/SL placement failed: {exc}",
                    file=sys.stderr,
                )
                pos.status = PositionStatus.OPEN  # still open, just no TP/SL

        elif updated.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
            pos.status = PositionStatus.FAILED
            pos.entry_order = updated
            self._dirty = True
            print(
                f"[{pos.position_id}] Entry order {updated.status.value} for {pos.signal.ticker}.",
                file=sys.stderr,
            )

    # -- Exit fill detection ---------------------------------------------------

    def _query_order(self, ticker: str, order: ExchangeOrder) -> ExchangeOrder:
        """Query order status, routing to algo API when needed."""
        if order.algo_id > 0:
            return self._client.get_algo_order(order.algo_id)
        return self._client.get_order(ticker, order.order_id)

    def _cancel_order_safe(self, ticker: str, order: ExchangeOrder) -> None:
        """Cancel order, routing to algo API when needed."""
        try:
            if order.algo_id > 0:
                self._client.cancel_algo_order(order.algo_id)
            else:
                self._client.cancel_order(ticker, order.order_id)
        except Exception:
            pass  # may already be canceled

    def _check_exit_fills(self, pos: LivePosition) -> None:
        tp_filled = False
        sl_filled = False

        if pos.tp_order is not None:
            try:
                updated_tp = self._query_order(pos.signal.ticker, pos.tp_order)
                pos.tp_order = updated_tp
                tp_filled = updated_tp.status is OrderStatus.FILLED
            except Exception as exc:
                print(
                    f"[{pos.position_id}] Failed to query TP order: {exc}",
                    file=sys.stderr,
                )
                return

        if pos.sl_order is not None:
            try:
                updated_sl = self._query_order(pos.signal.ticker, pos.sl_order)
                pos.sl_order = updated_sl
                sl_filled = updated_sl.status is OrderStatus.FILLED
            except Exception as exc:
                print(
                    f"[{pos.position_id}] Failed to query SL order: {exc}",
                    file=sys.stderr,
                )
                return

        if not tp_filled and not sl_filled:
            return

        # One side filled — cancel the other
        if tp_filled and pos.sl_order is not None and pos.sl_order.status is OrderStatus.NEW:
            self._cancel_order_safe(pos.signal.ticker, pos.sl_order)

        if sl_filled and pos.tp_order is not None and pos.tp_order.status is OrderStatus.NEW:
            self._cancel_order_safe(pos.signal.ticker, pos.tp_order)

        # Compute PnL
        if tp_filled:
            exit_price = pos.tp_order.stop_price if pos.tp_order else pos.fill_price
        else:
            exit_price = pos.sl_order.stop_price if pos.sl_order else pos.fill_price

        net_pnl, gross_pnl, fee_drag = compute_pnl(
            pos.fill_price,
            exit_price,
            pos.signal.position_type,
            leverage=pos.signal.leverage,
            taker_fee_rate=pos.signal.taker_fee_rate,
        )
        pos.pnl_pct = net_pnl
        pos.gross_pnl_pct = gross_pnl
        pos.fee_drag_pct = fee_drag
        pos.status = PositionStatus.CLOSED
        pos.closed_at = datetime.now(UTC)
        self._dirty = True

        exit_reason = "TP" if tp_filled else "SL"
        print(
            f"[{pos.position_id}] {pos.signal.ticker} closed via {exit_reason} "
            f"@ {exit_price:.4f} | PnL: {net_pnl:+.2f}%",
            file=sys.stderr,
        )
        try:
            balance = self._client.get_available_balance()
            print(f"Available capital: {balance:.2f} USDT", file=sys.stderr)
        except Exception:
            pass

    # -- State persistence -----------------------------------------------------

    def save_state(self, force: bool = False) -> None:
        """Persist active positions to disk for crash recovery."""
        if not force and not self._dirty:
            return
        active = [p for p in self._positions if p.status in (
            PositionStatus.PENDING_ENTRY, PositionStatus.OPEN,
        )]
        data = [self._serialize_position(p) for p in active]
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(data, indent=2, default=str))
        self._dirty = False

    def load_state(self) -> None:
        """Load positions from disk and reconcile with exchange."""
        if not _STATE_PATH.exists():
            return
        try:
            data = json.loads(_STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for item in data:
            pos = self._deserialize_position(item)
            if pos is not None:
                self._positions.append(pos)
                print(
                    f"[{pos.position_id}] Recovered position {pos.signal.ticker} "
                    f"status={pos.status.value}",
                    file=sys.stderr,
                )

    # -- Serialization ---------------------------------------------------------

    @staticmethod
    def _serialize_position(pos: LivePosition) -> dict[str, Any]:
        def _order_dict(order: ExchangeOrder | None) -> dict[str, Any] | None:
            if order is None:
                return None
            return {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "stop_price": order.stop_price,
                "status": order.status.value,
                "filled_qty": order.filled_qty,
                "avg_fill_price": order.avg_fill_price,
                "algo_id": order.algo_id,
            }

        def _signal_dict(sig: Signal) -> dict[str, Any]:
            return {
                "signal_date": sig.signal_date.isoformat(),
                "position_type": sig.position_type.value,
                "ticker": sig.ticker,
                "tp_pct": sig.tp_pct,
                "sl_pct": sig.sl_pct,
                "tp_price": sig.tp_price,
                "sl_price": sig.sl_price,
                "leverage": sig.leverage,
                "market_type": sig.market_type.value,
                "taker_fee_rate": sig.taker_fee_rate,
                "entry_price": sig.entry_price,
                "fill_timeout_seconds": sig.fill_timeout_seconds,
                "entry_delay_seconds": sig.entry_delay_seconds,
            }

        return {
            "position_id": pos.position_id,
            "status": pos.status.value,
            "signal": _signal_dict(pos.signal),
            "entry_order": _order_dict(pos.entry_order),
            "tp_order": _order_dict(pos.tp_order),
            "sl_order": _order_dict(pos.sl_order),
            "fill_price": pos.fill_price,
            "quantity": pos.quantity,
        }

    @staticmethod
    def _deserialize_position(data: dict[str, Any]) -> LivePosition | None:
        def _parse_order(d: dict[str, Any] | None) -> ExchangeOrder | None:
            if d is None:
                return None
            return ExchangeOrder(
                order_id=d["order_id"],
                symbol=d["symbol"],
                side=OrderSide(d["side"]),
                order_type=OrderType(d["order_type"]),
                quantity=d["quantity"],
                price=d["price"],
                stop_price=d["stop_price"],
                status=OrderStatus(d["status"]),
                filled_qty=d.get("filled_qty", 0),
                avg_fill_price=d.get("avg_fill_price", 0),
                algo_id=d.get("algo_id", 0),
            )

        try:
            sig_data = data["signal"]
            signal = Signal(
                signal_date=datetime.fromisoformat(sig_data["signal_date"]),
                position_type=PositionType(sig_data["position_type"]),
                ticker=sig_data["ticker"],
                tp_pct=sig_data.get("tp_pct"),
                sl_pct=sig_data.get("sl_pct"),
                tp_price=sig_data.get("tp_price"),
                sl_price=sig_data.get("sl_price"),
                leverage=sig_data.get("leverage", 1.0),
                taker_fee_rate=sig_data.get("taker_fee_rate", 0.0005),
                entry_price=sig_data.get("entry_price"),
                fill_timeout_seconds=sig_data.get("fill_timeout_seconds", 3600),
                entry_delay_seconds=sig_data.get("entry_delay_seconds", 5),
            )
            return LivePosition(
                signal=signal,
                position_id=data["position_id"],
                status=PositionStatus(data["status"]),
                entry_order=_parse_order(data.get("entry_order")),
                tp_order=_parse_order(data.get("tp_order")),
                sl_order=_parse_order(data.get("sl_order")),
                fill_price=data.get("fill_price", 0),
                quantity=data.get("quantity", 0),
            )
        except (KeyError, ValueError) as exc:
            print(f"Failed to deserialize position: {exc}", file=sys.stderr)
            return None
