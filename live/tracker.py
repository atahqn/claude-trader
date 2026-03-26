from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backtester.data import _symbol_for_api
from backtester.models import PositionType, Signal
from backtester.resolver import compute_pnl

from .auth_client import BinanceFuturesClient
from .executor import OrderExecutor
from .models import (
    ExchangeOrder,
    LiveConfig,
    LivePosition,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
)

_STATE_DIR = Path.home() / ".claude_trader"
_STATE_PATH = _STATE_DIR / "live_state.json"
_POSITION_AMOUNT_EPSILON = 0.5


class PositionTracker:
    """Manages position lifecycle: PENDING_ENTRY → OPEN → CLOSED / FAILED."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        executor: OrderExecutor,
        config: LiveConfig,
    ) -> None:
        self._client = client
        self._executor = executor
        self._config = config
        self._positions: list[LivePosition] = []
        self._external_position_keys: set[tuple[str, str]] = set()
        self._dirty = False

    @property
    def positions(self) -> list[LivePosition]:
        return self._positions

    @property
    def open_count(self) -> int:
        tracked_open = sum(
            1 for p in self._positions
            if p.status in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN)
        )
        return tracked_open + len(self._external_position_keys)

    # -- Position management ---------------------------------------------------

    def add_position(self, position: LivePosition) -> None:
        self._positions.append(position)
        self._dirty = True

    def reconcile_with_exchange(self) -> None:
        """Track exchange positions that are not present in local state."""
        try:
            raw_positions = self._client.get_position_info()
        except Exception as exc:
            print(f"Failed to reconcile exchange positions: {exc}", file=sys.stderr)
            return

        tracked_keys = self._tracked_exchange_keys()
        external_keys = self._exchange_position_keys(raw_positions) - tracked_keys
        if external_keys == self._external_position_keys:
            return

        self._external_position_keys = external_keys
        if external_keys:
            rendered = ", ".join(f"{symbol}:{side}" for symbol, side in sorted(external_keys))
            print(
                f"Recovered {len(external_keys)} untracked exchange position(s): {rendered}",
                file=sys.stderr,
            )
        else:
            print("No untracked exchange positions remain.", file=sys.stderr)

    def has_external_conflict(self, signal: Signal) -> bool:
        api_symbol = _symbol_for_api(signal.ticker)
        return any(symbol == api_symbol for symbol, _side in self._external_position_keys)

    def check_fills(self, now_utc: datetime | None = None) -> None:
        """Poll exchange for order status updates on all active positions."""
        now_utc = now_utc or self._client.server_now()
        for pos in self._positions:
            if pos.status is PositionStatus.PENDING_ENTRY:
                self._check_entry_fill(pos, now_utc)
            elif pos.status is PositionStatus.OPEN:
                exit_closed = self._check_exit_fills(pos, now_utc)
                if exit_closed is not False:
                    continue
                self._check_timeout(pos, now_utc)

    # -- Entry fill detection --------------------------------------------------

    def _check_entry_fill(self, pos: LivePosition, now_utc: datetime) -> None:
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
            age = (now_utc - updated.created_at).total_seconds()
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
            pos.opened_at = updated.updated_at or updated.created_at or now_utc
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

    def _tracked_exchange_keys(self) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for pos in self._positions:
            if pos.status not in (PositionStatus.PENDING_ENTRY, PositionStatus.OPEN):
                continue
            side = "LONG" if pos.signal.position_type is PositionType.LONG else "SHORT"
            keys.add((_symbol_for_api(pos.signal.ticker), side))
        return keys

    @staticmethod
    def _exchange_position_keys(raw_positions: list[dict[str, Any]]) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for row in raw_positions:
            symbol = str(row.get("symbol", ""))
            if not symbol:
                continue
            try:
                amount = float(row.get("positionAmt", 0))
            except (TypeError, ValueError):
                continue
            if abs(amount) <= _POSITION_AMOUNT_EPSILON:
                continue
            side = str(row.get("positionSide", "BOTH"))
            if side not in {"LONG", "SHORT"}:
                side = "LONG" if amount > 0 else "SHORT"
            keys.add((symbol, side))
        return keys

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

    @staticmethod
    def _position_side(pos: LivePosition) -> str:
        return "LONG" if pos.signal.position_type is PositionType.LONG else "SHORT"

    def _has_open_exchange_position(self, pos: LivePosition) -> bool | None:
        """Return whether the exchange still reports an open position for *pos*."""
        try:
            raw_positions = self._client.get_position_info(pos.signal.ticker)
        except Exception as exc:
            print(
                f"[{pos.position_id}] Failed to query exchange position: {exc}",
                file=sys.stderr,
            )
            return None

        api_symbol = _symbol_for_api(pos.signal.ticker)
        target_side = self._position_side(pos)
        for row in raw_positions:
            if str(row.get("symbol", "")) != api_symbol:
                continue
            try:
                amount = float(row.get("positionAmt", 0))
            except (TypeError, ValueError):
                continue
            if abs(amount) <= _POSITION_AMOUNT_EPSILON:
                continue

            side = str(row.get("positionSide", "BOTH"))
            if side in {"LONG", "SHORT"}:
                if side == target_side:
                    return True
                continue
            if target_side == "LONG" and amount > _POSITION_AMOUNT_EPSILON:
                return True
            if target_side == "SHORT" and amount < -_POSITION_AMOUNT_EPSILON:
                return True

        return False

    @staticmethod
    def _infer_exchange_exit(pos: LivePosition) -> tuple[str, ExchangeOrder | None]:
        canceled_statuses = {
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
        tp_canceled = (
            pos.tp_order is not None
            and pos.tp_order.status in canceled_statuses
        )
        sl_canceled = (
            pos.sl_order is not None
            and pos.sl_order.status in canceled_statuses
        )

        if tp_canceled and sl_canceled:
            return "EXTERNAL", None
        if sl_canceled and pos.tp_order is not None:
            return "TP", pos.tp_order
        if tp_canceled and pos.sl_order is not None:
            return "SL", pos.sl_order
        if pos.tp_order is not None and pos.tp_order.avg_fill_price > 0:
            return "TP", pos.tp_order
        if pos.sl_order is not None and pos.sl_order.avg_fill_price > 0:
            return "SL", pos.sl_order
        return "EXTERNAL", pos.tp_order or pos.sl_order

    def _check_exit_fills(self, pos: LivePosition, now_utc: datetime) -> bool | None:
        tp_filled = False
        sl_filled = False
        query_failed = False

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
                query_failed = True

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
                query_failed = True

        if tp_filled or sl_filled:
            # One side filled — cancel the other
            if tp_filled and pos.sl_order is not None and pos.sl_order.status is OrderStatus.NEW:
                self._cancel_order_safe(pos.signal.ticker, pos.sl_order)

            if sl_filled and pos.tp_order is not None and pos.tp_order.status is OrderStatus.NEW:
                self._cancel_order_safe(pos.signal.ticker, pos.tp_order)

            exit_reason = "TP" if tp_filled else "SL"
            exit_order = pos.tp_order if tp_filled else pos.sl_order
            self._finalize_close(pos, exit_order, exit_reason, now_utc)
            return True

        exchange_open = self._has_open_exchange_position(pos)
        if exchange_open is False:
            exit_reason, exit_order = self._infer_exchange_exit(pos)
            self._finalize_close(pos, exit_order, exit_reason, now_utc)
            return True

        if query_failed or exchange_open is None:
            return None

        return False

    def _check_timeout(self, pos: LivePosition, now_utc: datetime) -> bool:
        if pos.opened_at is None:
            return False

        holding_hours = pos.signal.max_holding_hours
        if holding_hours is None:
            holding_hours = self._config.max_holding_hours

        deadline = pos.opened_at + timedelta(hours=holding_hours)
        if now_utc < deadline:
            return False

        if pos.tp_order is not None and pos.tp_order.status is OrderStatus.NEW:
            self._cancel_order_safe(pos.signal.ticker, pos.tp_order)
        if pos.sl_order is not None and pos.sl_order.status is OrderStatus.NEW:
            self._cancel_order_safe(pos.signal.ticker, pos.sl_order)

        try:
            exit_order = self._executor.close_position_market(pos)
        except Exception as exc:
            print(
                f"[{pos.position_id}] Timeout close failed for {pos.signal.ticker}: {exc}",
                file=sys.stderr,
            )
            return False

        self._finalize_close(pos, exit_order, "TIMEOUT", now_utc)
        return True

    def _finalize_close(
        self,
        pos: LivePosition,
        exit_order: ExchangeOrder | None,
        exit_reason: str,
        now_utc: datetime,
    ) -> None:
        exit_price = pos.fill_price
        if exit_order is not None:
            if exit_order.avg_fill_price > 0:
                exit_price = exit_order.avg_fill_price
            elif exit_order.stop_price > 0:
                exit_price = exit_order.stop_price
            elif exit_order.price > 0:
                exit_price = exit_order.price

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
        pos.closed_at = (
            exit_order.updated_at if exit_order is not None and exit_order.updated_at is not None
            else exit_order.created_at if exit_order is not None and exit_order.created_at is not None
            else now_utc
        )
        self._dirty = True

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
                "max_holding_hours": sig.max_holding_hours,
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
            "opened_at": pos.opened_at.isoformat() if pos.opened_at is not None else None,
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
                max_holding_hours=sig_data.get("max_holding_hours"),
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
                opened_at=(
                    datetime.fromisoformat(data["opened_at"])
                    if data.get("opened_at") is not None else None
                ),
            )
        except (KeyError, ValueError) as exc:
            print(f"Failed to deserialize position: {exc}", file=sys.stderr)
            return None
