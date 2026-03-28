from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backtester.data import (
    BybitClient,
    _RateLimiter,
    _retry_delay,
    _symbol_for_api,
    _to_millis,
)

from .models import (
    AccountTrade,
    ExchangeOrder,
    LiveConfig,
    OrderSide,
    OrderStatus,
    OrderType,
)

_DEFAULT_RETRY_DELAY = 15.0
_MAX_RETRY_DELAY = 300.0
_RETRYABLE_STATUS_CODES = {418, 429}
_RECV_WINDOW = 5000
_MAX_TIMESTAMP_RETRIES = 1
_CLOCK_JUMP_THRESHOLD_MS = 1000
_TIME_SYNC_INTERVAL = 300  # re-sync every 5 minutes

_ORDER_STATUS_MAP = {
    "New": OrderStatus.NEW,
    "Created": OrderStatus.NEW,
    "PartiallyFilled": OrderStatus.NEW,
    "Untriggered": OrderStatus.NEW,
    "Triggered": OrderStatus.NEW,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELED,
    "Deactivated": OrderStatus.CANCELED,
    "Rejected": OrderStatus.REJECTED,
}


def _from_millis(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _normalize_side(raw_side: str) -> OrderSide:
    return OrderSide(raw_side.upper())


def _normalize_order_type(raw: dict[str, Any]) -> OrderType:
    order_type = str(raw.get("orderType", raw.get("stopOrderType", "MARKET"))).upper()
    if order_type == "MARKET":
        stop_type = str(raw.get("stopOrderType", "")).upper()
        if stop_type == "STOP":
            return OrderType.STOP_MARKET
        if stop_type in {"TAKE_PROFIT", "TAKEPROFIT"}:
            return OrderType.TAKE_PROFIT_MARKET
    try:
        return OrderType(order_type)
    except ValueError:
        return OrderType.MARKET


def _parse_order(raw: dict[str, Any]) -> ExchangeOrder:
    created_ms = raw.get("createdTime")
    updated_ms = raw.get("updatedTime")
    trigger_price = raw.get("triggerPrice", raw.get("stopPx", 0))
    qty = raw.get("qty", raw.get("cumExecQty", 0))
    status = _ORDER_STATUS_MAP.get(str(raw.get("orderStatus", "New")), OrderStatus.NEW)
    return ExchangeOrder(
        order_id=str(raw.get("orderId", "")),
        symbol=str(raw["symbol"]),
        side=_normalize_side(str(raw["side"])),
        order_type=_normalize_order_type(raw),
        quantity=float(qty or 0),
        price=float(raw.get("price", 0) or 0),
        stop_price=float(trigger_price or 0),
        status=status,
        filled_qty=float(raw.get("cumExecQty", 0) or 0),
        avg_fill_price=float(raw.get("avgPrice", 0) or 0),
        created_at=_from_millis(int(created_ms)) if created_ms not in (None, "") else None,
        updated_at=_from_millis(int(updated_ms)) if updated_ms not in (None, "") else None,
        is_conditional=str(raw.get("triggerPrice", "")) not in {"", "0", "0.0"},
    )


def _parse_account_trade(raw: dict[str, Any]) -> AccountTrade:
    position_idx = str(raw.get("positionIdx", "0"))
    if position_idx == "1":
        position_side = "LONG"
    elif position_idx == "2":
        position_side = "SHORT"
    else:
        position_side = "BOTH"
    return AccountTrade(
        trade_id=str(raw.get("execId", "")),
        order_id=str(raw.get("orderId", "")),
        symbol=str(raw["symbol"]),
        side=_normalize_side(str(raw["side"])),
        price=float(raw.get("execPrice", 0) or 0),
        quantity=float(raw.get("execQty", 0) or 0),
        time=_from_millis(int(raw["execTime"])),
        realized_pnl=float(raw.get("closedPnl", 0) or 0),
        commission=float(raw.get("execFee", 0) or 0),
        commission_asset=str(raw.get("feeCurrency", "")),
        position_side=position_side,
    )


def _position_idx(position_side: str) -> int:
    if position_side == "LONG":
        return 1
    if position_side == "SHORT":
        return 2
    return 0


def _trigger_direction(order_type: OrderType, side: OrderSide) -> int:
    if order_type is OrderType.TAKE_PROFIT_MARKET:
        return 1 if side is OrderSide.SELL else 2
    return 2 if side is OrderSide.SELL else 1


class BybitFuturesClient:
    """Authenticated Bybit V5 linear futures REST client (stdlib-only)."""

    def __init__(self, config: LiveConfig) -> None:
        self._api_key = config.api_key
        self._api_secret = config.api_secret.encode()
        self._base_url = config.base_url
        self._rate_limiter = _RateLimiter(limit_per_minute=600)
        self._time_offset_ms: int = 0
        self._last_time_sync_monotonic: float = 0.0
        self._last_sync_wall_ms: int = 0
        self._last_sync_monotonic_ms: int = 0
        self._sync_server_time()

    def _sync_server_time(self) -> None:
        try:
            local_before = int(time.time() * 1000)
            url = f"{self._base_url}/v5/market/time"
            with urlopen(url) as response:
                payload = json.loads(response.read())
            server_time = int(payload["result"]["timeSecond"]) * 1000
            local_after = int(time.time() * 1000)
            local_mid = (local_before + local_after) // 2
            self._time_offset_ms = server_time - local_mid
            self._last_time_sync_monotonic = time.monotonic()
            self._last_sync_wall_ms = int(time.time() * 1000)
            self._last_sync_monotonic_ms = int(time.monotonic() * 1000)
            if abs(self._time_offset_ms) > 500:
                print(
                    f"Clock offset: {self._time_offset_ms:+d}ms (synced with Bybit)",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"Failed to sync server time: {exc}", file=sys.stderr)

    def _clock_jump_detected(self) -> bool:
        if self._last_sync_monotonic_ms <= 0:
            return False
        wall_delta = int(time.time() * 1000) - self._last_sync_wall_ms
        monotonic_delta = int(time.monotonic() * 1000) - self._last_sync_monotonic_ms
        return abs(wall_delta - monotonic_delta) > _CLOCK_JUMP_THRESHOLD_MS

    def _ensure_time_sync(self, *, force: bool = False) -> None:
        if force or self._last_time_sync_monotonic <= 0:
            self._sync_server_time()
            return
        if time.monotonic() - self._last_time_sync_monotonic > _TIME_SYNC_INTERVAL:
            self._sync_server_time()
            return
        if self._clock_jump_detected():
            print("Detected local clock jump; re-syncing with Bybit.", file=sys.stderr)
            self._sync_server_time()

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def server_now(self) -> datetime:
        self._ensure_time_sync()
        return datetime.fromtimestamp(self._timestamp_ms() / 1000, tz=UTC)

    def _sign(self, payload: str) -> dict[str, str]:
        self._ensure_time_sync()
        timestamp = str(self._timestamp_ms())
        recv_window = str(_RECV_WINDOW)
        prehash = f"{timestamp}{self._api_key}{recv_window}{payload}"
        signature = hmac.new(self._api_secret, prehash.encode(), hashlib.sha256).hexdigest()
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }

    @staticmethod
    def _read_error_body(exc: HTTPError) -> str:
        try:
            return exc.read().decode()
        except Exception:
            return ""

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        weight: int = 1,
    ) -> Any:
        params = params or {}
        next_delay = _DEFAULT_RETRY_DELAY
        attempt = 1
        timestamp_retry_count = 0

        while True:
            query = urlencode(params)
            body_text = json.dumps(params, separators=(",", ":"), sort_keys=True) if method != "GET" else ""
            payload_for_signing = query if method == "GET" else body_text
            headers: dict[str, str] = {}
            if signed:
                headers.update(self._sign(payload_for_signing))

            if method == "GET":
                url = f"{self._base_url}{path}"
                if query:
                    url = f"{url}?{query}"
                body = None
            else:
                url = f"{self._base_url}{path}"
                body = body_text.encode()
                headers["Content-Type"] = "application/json"

            self._rate_limiter.acquire(weight)
            req = Request(url, data=body, headers=headers, method=method)
            try:
                with urlopen(req) as response:
                    payload = json.loads(response.read())
                    if payload.get("retCode", 0) != 0:
                        raise RuntimeError(
                            f"Bybit API error {payload.get('retCode')} on {method} {path}: "
                            f"{payload.get('retMsg', 'unknown error')}"
                        )
                    return payload.get("result", {})
            except HTTPError as exc:
                error_body = self._read_error_body(exc)
                if exc.code not in _RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"Bybit API error {exc.code} on {method} {path}: {error_body}"
                    ) from exc
                delay = _retry_delay(exc, next_delay)
                print(
                    f"Bybit throttled ({exc.code}) on {path}; "
                    f"waiting {delay:.1f}s before retry {attempt + 1}.",
                    file=sys.stderr,
                )
                time.sleep(delay)
                next_delay = min(max(delay * 2, _DEFAULT_RETRY_DELAY), _MAX_RETRY_DELAY)
                attempt += 1
            except RuntimeError as exc:
                if signed and "10002" in str(exc) and timestamp_retry_count < _MAX_TIMESTAMP_RETRIES:
                    timestamp_retry_count += 1
                    print(
                        f"Bybit rejected timestamp on {path}; re-syncing server time and retrying.",
                        file=sys.stderr,
                    )
                    self._ensure_time_sync(force=True)
                    continue
                raise

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        position_side: str = "BOTH",
    ) -> ExchangeOrder:
        params = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "side": side.value.title(),
            "orderType": "Market",
            "qty": str(quantity),
            "positionIdx": _position_idx(position_side),
        }
        raw = self._request("POST", "/v5/order/create", params, weight=1)
        return ExchangeOrder(
            order_id=str(raw.get("orderId", "")),
            symbol=_symbol_for_api(symbol),
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0.0,
            stop_price=0.0,
            status=OrderStatus.NEW,
        )

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        position_side: str = "BOTH",
    ) -> ExchangeOrder:
        params = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "side": side.value.title(),
            "orderType": "Limit",
            "timeInForce": "GTC",
            "qty": str(quantity),
            "price": str(price),
            "positionIdx": _position_idx(position_side),
        }
        raw = self._request("POST", "/v5/order/create", params, weight=1)
        return ExchangeOrder(
            order_id=str(raw.get("orderId", "")),
            symbol=_symbol_for_api(symbol),
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            stop_price=0.0,
            status=OrderStatus.NEW,
        )

    def _place_conditional_market(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        *,
        order_type: OrderType,
        position_side: str = "BOTH",
        quantity: float | None = None,
    ) -> ExchangeOrder:
        params = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "side": side.value.title(),
            "orderType": "Market",
            "qty": str(quantity if quantity is not None else 0),
            "triggerPrice": str(stop_price),
            "triggerDirection": _trigger_direction(order_type, side),
            "triggerBy": "MarkPrice",
            "positionIdx": _position_idx(position_side),
            "reduceOnly": True,
            "closeOnTrigger": True,
            "orderFilter": "StopOrder",
            "stopOrderType": "TakeProfit" if order_type is OrderType.TAKE_PROFIT_MARKET else "Stop",
        }
        raw = self._request("POST", "/v5/order/create", params, weight=1)
        return ExchangeOrder(
            order_id=str(raw.get("orderId", "")),
            symbol=_symbol_for_api(symbol),
            side=side,
            order_type=order_type,
            quantity=quantity or 0.0,
            price=0.0,
            stop_price=stop_price,
            status=OrderStatus.NEW,
            is_conditional=True,
        )

    def place_stop_market(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        position_side: str = "BOTH",
        quantity: float | None = None,
    ) -> ExchangeOrder:
        return self._place_conditional_market(
            symbol,
            side,
            stop_price,
            order_type=OrderType.STOP_MARKET,
            position_side=position_side,
            quantity=quantity,
        )

    def place_take_profit_market(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        position_side: str = "BOTH",
        quantity: float | None = None,
    ) -> ExchangeOrder:
        return self._place_conditional_market(
            symbol,
            side,
            stop_price,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            position_side=position_side,
            quantity=quantity,
        )

    def cancel_order(
        self,
        symbol: str,
        order_id: str,
        *,
        conditional: bool = False,
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "orderId": order_id,
        }
        if conditional:
            params["orderFilter"] = "StopOrder"
        self._request("POST", "/v5/order/cancel", params, weight=1)
        return self.get_order(symbol, order_id, conditional=conditional)

    def get_order(
        self,
        symbol: str,
        order_id: str,
        *,
        conditional: bool = False,
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "orderId": order_id,
        }
        if conditional:
            params["orderFilter"] = "StopOrder"
        raw = self._request("GET", "/v5/order/realtime", params, weight=1)
        rows = raw.get("list", [])
        if not rows:
            history = self._request("GET", "/v5/order/history", params, weight=1)
            rows = history.get("list", [])
        if not rows:
            raise RuntimeError(f"Order not found: {order_id}")
        return _parse_order(rows[0])

    def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        params: dict[str, Any] = {"category": "linear"}
        if symbol is not None:
            params["symbol"] = _symbol_for_api(symbol)
        raw = self._request("GET", "/v5/order/realtime", params, weight=1)
        return [_parse_order(row) for row in raw.get("list", [])]

    def get_position_info(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol is not None:
            params["symbol"] = _symbol_for_api(symbol)
        raw = self._request("GET", "/v5/position/list", params, weight=5)
        return raw.get("list", [])

    def get_account_trades(
        self,
        symbol: str,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> list[AccountTrade]:
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "limit": max(1, min(limit, 100)),
        }
        if start_time is not None:
            params["startTime"] = _to_millis(start_time)
        if end_time is not None:
            params["endTime"] = _to_millis(end_time)
        if order_id:
            params["orderId"] = order_id
        raw = self._request("GET", "/v5/execution/list", params, weight=5)
        return [_parse_account_trade(row) for row in raw.get("list", [])]

    def get_account_info(self) -> dict[str, Any]:
        return self._request(
            "GET",
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED"},
            weight=5,
        )

    def get_available_balance(self) -> float:
        info = self.get_account_info()
        accounts = info.get("list", [])
        if not accounts:
            return 0.0
        return float(accounts[0].get("totalAvailableBalance", 0) or 0)

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        params = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        try:
            return self._request("POST", "/v5/position/set-leverage", params, weight=1)
        except RuntimeError as exc:
            if "110043" in str(exc):
                return {}
            raise

    def get_exchange_info(self) -> dict[str, Any]:
        symbols: list[dict[str, Any]] = []
        cursor = ""
        while True:
            params: dict[str, Any] = {
                "category": "linear",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            raw = self._request("GET", "/v5/market/instruments-info", params, signed=False, weight=1)
            symbols.extend(raw.get("list", []))
            cursor = str(raw.get("nextPageCursor", "") or "")
            if not cursor:
                break
        return {"symbols": symbols}

    def get_mark_price(self, symbol: str) -> float:
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": _symbol_for_api(symbol),
        }
        raw = self._request("GET", "/v5/market/tickers", params, signed=False, weight=1)
        rows = raw.get("list", [])
        if not rows:
            raise RuntimeError(f"Mark price unavailable for {symbol}")
        return float(rows[0]["markPrice"])


class LiveMarketClient(BybitClient):
    """No-cache subclass of BybitClient for live market data."""

    def __init__(self, config: LiveConfig) -> None:
        super().__init__(base_url=config.base_url)

    def _get_json(self, path: str, params: dict[str, Any], *, weight: int) -> Any:
        query = urlencode(params)
        url = f"{self._base_url}{path}?{query}"
        next_delay = _DEFAULT_RETRY_DELAY
        attempt = 1

        while True:
            self._rate_limiter.acquire(weight)
            try:
                with urlopen(url) as response:
                    payload = json.loads(response.read())
                    if payload.get("retCode", 0) != 0:
                        raise RuntimeError(
                            f"Bybit API error {payload.get('retCode')} on GET {path}: "
                            f"{payload.get('retMsg', 'unknown error')}"
                        )
                    return payload.get("result", {})
            except HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS_CODES:
                    raise
                delay = _retry_delay(exc, next_delay)
                symbol = params.get("symbol", "")
                print(
                    f"Bybit throttled ({exc.code}) on {path} symbol={symbol}; "
                    f"waiting {delay:.1f}s before retry {attempt + 1}.",
                    file=sys.stderr,
                )
                time.sleep(delay)
                next_delay = min(max(delay * 2, _DEFAULT_RETRY_DELAY), _MAX_RETRY_DELAY)
                attempt += 1
