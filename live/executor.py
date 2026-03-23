from __future__ import annotations

import uuid
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any

from backtester.data import _symbol_for_api
from backtester.models import PositionType, Signal
from backtester.resolver import compute_tp_sl_prices

from .auth_client import BinanceFuturesClient
from .models import (
    ExchangeOrder,
    LiveConfig,
    LivePosition,
    OrderSide,
    OrderStatus,
    PositionStatus,
)


class OrderExecutor:
    """Converts a Signal into exchange orders."""

    def __init__(self, client: BinanceFuturesClient, config: LiveConfig) -> None:
        self._client = client
        self._config = config
        self._symbol_info: dict[str, dict[str, Any]] = {}

    # -- Public ----------------------------------------------------------------

    def execute_signal(self, signal: Signal) -> LivePosition:
        """Place entry order for *signal* and return a PENDING_ENTRY position."""
        api_symbol = _symbol_for_api(signal.ticker)

        # Ensure symbol info is cached for precision rounding
        if api_symbol not in self._symbol_info:
            self._load_symbol_info(api_symbol)

        # Set leverage
        self._client.set_leverage(signal.ticker, int(signal.leverage))

        # Determine entry price for quantity calculation
        if signal.entry_price is not None:
            entry_price = signal.entry_price
        else:
            entry_price = self._client.get_mark_price(signal.ticker)

        quantity, required_notional = self._compute_entry_quantity(
            api_symbol,
            entry_price,
            use_market_filter=signal.entry_price is None,
        )
        if required_notional > self._config.position_size_usdt + 1e-9:
            available_balance = self._client.get_available_balance()
            if available_balance + 1e-9 < required_notional:
                raise ValueError(
                    f"{signal.ticker} requires {required_notional:.2f} USDT after "
                    f"exchange min-notional rounding, but only "
                    f"{available_balance:.2f} USDT is available"
                )

        # Determine order side and position side (hedge mode)
        side = OrderSide.BUY if signal.position_type is PositionType.LONG else OrderSide.SELL
        position_side = "LONG" if signal.position_type is PositionType.LONG else "SHORT"

        # Place entry order
        if signal.entry_price is not None:
            price = self._round_price(api_symbol, signal.entry_price)
            entry_order = self._client.place_limit_order(
                signal.ticker, side, quantity, price, position_side,
            )
        else:
            entry_order = self._client.place_market_order(
                signal.ticker, side, quantity, position_side,
            )

        return LivePosition(
            signal=signal,
            position_id=uuid.uuid4().hex[:12],
            status=PositionStatus.PENDING_ENTRY,
            entry_order=entry_order,
            quantity=quantity,
        )

    def place_tp_sl(self, position: LivePosition) -> None:
        """Place TP and SL orders on the exchange using the actual fill price."""
        signal = position.signal
        fill_price = position.fill_price

        api_symbol = _symbol_for_api(signal.ticker)
        if api_symbol not in self._symbol_info:
            self._load_symbol_info(api_symbol)

        # Compute TP/SL prices using the same logic as backtest
        tp_price, sl_price = compute_tp_sl_prices(
            fill_price,
            signal.position_type,
            tp_pct=signal.tp_pct,
            sl_pct=signal.sl_pct,
            taker_fee_rate=signal.taker_fee_rate,
            tp_price_override=signal.tp_price,
            sl_price_override=signal.sl_price,
        )

        tp_price = self._round_price(api_symbol, tp_price)
        sl_price = self._round_price(api_symbol, sl_price)

        # Close side is opposite of entry side; positionSide stays the same
        close_side = (
            OrderSide.SELL if signal.position_type is PositionType.LONG else OrderSide.BUY
        )
        position_side = "LONG" if signal.position_type is PositionType.LONG else "SHORT"

        position.tp_order = self._client.place_take_profit_market(
            signal.ticker, close_side, tp_price, position_side,
        )
        position.sl_order = self._client.place_stop_market(
            signal.ticker, close_side, sl_price, position_side,
        )

    def close_position_market(self, position: LivePosition) -> ExchangeOrder:
        """Close an open position immediately with an opposing market order."""
        signal = position.signal
        api_symbol = _symbol_for_api(signal.ticker)
        if api_symbol not in self._symbol_info:
            self._load_symbol_info(api_symbol)

        quantity = self._round_quantity(
            api_symbol,
            position.quantity,
            use_market_filter=True,
        )
        if quantity <= 0:
            raise ValueError(f"Cannot close {signal.ticker}: rounded quantity is zero")

        close_side = (
            OrderSide.SELL if signal.position_type is PositionType.LONG else OrderSide.BUY
        )
        position_side = "LONG" if signal.position_type is PositionType.LONG else "SHORT"

        exit_order = self._client.place_market_order(
            signal.ticker,
            close_side,
            quantity,
            position_side,
        )
        if exit_order.status is not OrderStatus.FILLED or exit_order.avg_fill_price <= 0:
            exit_order = self._client.get_order(signal.ticker, exit_order.order_id)
        return exit_order

    # -- Helpers ---------------------------------------------------------------

    def _load_symbol_info(self, api_symbol: str) -> None:
        info = self._client.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == api_symbol:
                self._symbol_info[api_symbol] = s
                return
        raise ValueError(f"Symbol {api_symbol} not found in exchange info")

    def _compute_entry_quantity(
        self,
        api_symbol: str,
        entry_price: float,
        *,
        use_market_filter: bool,
    ) -> tuple[float, float]:
        raw_qty = self._config.position_size_usdt / entry_price
        quantity = self._round_quantity(
            api_symbol,
            raw_qty,
            use_market_filter=use_market_filter,
        )

        min_qty = self._minimum_quantity(api_symbol, use_market_filter=use_market_filter)
        min_notional = self._minimum_notional(api_symbol)

        if min_qty > 0:
            quantity = max(quantity, min_qty)
        if min_notional > 0 and quantity * entry_price + 1e-9 < min_notional:
            quantity = max(quantity, min_notional / entry_price)

        quantity = self._round_quantity_up(
            api_symbol,
            quantity,
            use_market_filter=use_market_filter,
        )
        if quantity <= 0:
            raise ValueError(f"Cannot size {api_symbol}: rounded quantity is zero")

        required_notional = quantity * entry_price
        if min_notional > 0 and required_notional + 1e-9 < min_notional:
            raise ValueError(
                f"{api_symbol} requires at least {min_notional:.2f} USDT notional, "
                f"but rounded quantity {quantity} only yields {required_notional:.2f} USDT"
            )
        return quantity, required_notional

    def _round_quantity(
        self,
        api_symbol: str,
        qty: float,
        *,
        use_market_filter: bool = False,
    ) -> float:
        return self._round_quantity_with_mode(
            api_symbol,
            qty,
            use_market_filter=use_market_filter,
            rounding=ROUND_DOWN,
        )

    def _round_quantity_up(
        self,
        api_symbol: str,
        qty: float,
        *,
        use_market_filter: bool = False,
    ) -> float:
        return self._round_quantity_with_mode(
            api_symbol,
            qty,
            use_market_filter=use_market_filter,
            rounding=ROUND_UP,
        )

    def _round_quantity_with_mode(
        self,
        api_symbol: str,
        qty: float,
        *,
        use_market_filter: bool,
        rounding: str,
    ) -> float:
        info = self._symbol_info.get(api_symbol)
        if info is None:
            return qty

        qty_filter = self._quantity_filter(api_symbol, use_market_filter=use_market_filter)
        if qty_filter is None:
            return qty

        step = Decimal(str(qty_filter.get("stepSize", "0")))
        if step <= 0:
            return qty

        dec_qty = Decimal(str(qty))
        if dec_qty <= 0:
            return 0.0

        rounded = (dec_qty / step).to_integral_value(rounding=rounding) * step
        if rounded <= 0:
            return 0.0
        return float(rounded)

    def _quantity_filter(
        self,
        api_symbol: str,
        *,
        use_market_filter: bool,
    ) -> dict[str, Any] | None:
        info = self._symbol_info.get(api_symbol)
        if info is None:
            return None
        filters = info.get("filters", [])
        if use_market_filter:
            for f in filters:
                if f["filterType"] == "MARKET_LOT_SIZE":
                    return f
        for f in filters:
            if f["filterType"] == "LOT_SIZE":
                return f
        return None

    def _minimum_quantity(self, api_symbol: str, *, use_market_filter: bool) -> float:
        qty_filter = self._quantity_filter(api_symbol, use_market_filter=use_market_filter)
        if qty_filter is None:
            return 0.0
        return float(qty_filter.get("minQty", 0) or 0)

    def _minimum_notional(self, api_symbol: str) -> float:
        info = self._symbol_info.get(api_symbol)
        if info is None:
            return 0.0
        for f in info.get("filters", []):
            if f["filterType"] in {"MIN_NOTIONAL", "NOTIONAL"}:
                value = f.get("minNotional", f.get("notional", 0))
                return float(value or 0)
        return 0.0

    def _round_price(self, api_symbol: str, price: float) -> float:
        info = self._symbol_info.get(api_symbol)
        if info is None:
            return price
        for f in info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
                if tick > 0:
                    precision = len(f["tickSize"].rstrip("0").split(".")[-1])
                    return round(price - (price % tick), precision)
        return price
