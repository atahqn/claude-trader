from __future__ import annotations

import uuid
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

        # Compute quantity
        raw_qty = self._config.position_size_usdt / entry_price
        quantity = self._round_quantity(api_symbol, raw_qty)

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
            quantity=position.quantity,
        )
        position.sl_order = self._client.place_stop_market(
            signal.ticker, close_side, sl_price, position_side,
            quantity=position.quantity,
        )

    # -- Helpers ---------------------------------------------------------------

    def _load_symbol_info(self, api_symbol: str) -> None:
        info = self._client.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == api_symbol:
                self._symbol_info[api_symbol] = s
                return
        raise ValueError(f"Symbol {api_symbol} not found in exchange info")

    def _round_quantity(self, api_symbol: str, qty: float) -> float:
        info = self._symbol_info.get(api_symbol)
        if info is None:
            return qty
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                if step > 0:
                    precision = len(f["stepSize"].rstrip("0").split(".")[-1])
                    return round(qty - (qty % step), precision)
        return qty

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
