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
    BinanceClient,
    _RateLimiter,
    _retry_delay,
    _symbol_for_api,
    _sync_rate_limiter,
)

from .models import (
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


def _from_millis(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _parse_order(raw: dict[str, Any]) -> ExchangeOrder:
    return ExchangeOrder(
        order_id=int(raw["orderId"]),
        symbol=str(raw["symbol"]),
        side=OrderSide(raw["side"]),
        order_type=OrderType(raw.get("origType", raw.get("type", "MARKET"))),
        quantity=float(raw.get("origQty", 0)),
        price=float(raw.get("price", 0)),
        stop_price=float(raw.get("stopPrice", 0)),
        status=OrderStatus(raw["status"]),
        filled_qty=float(raw.get("executedQty", 0)),
        avg_fill_price=float(raw.get("avgPrice", 0)),
        created_at=_from_millis(int(raw["time"])) if "time" in raw else None,
        updated_at=_from_millis(int(raw["updateTime"])) if "updateTime" in raw else None,
    )


# Algo order statuses → our OrderStatus mapping
_ALGO_STATUS_MAP = {
    "NEW": OrderStatus.NEW,
    "TRIGGERED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "CANCELLED": OrderStatus.CANCELED,
    "EXPIRED": OrderStatus.EXPIRED,
    "REJECTED": OrderStatus.REJECTED,
}


def _parse_algo_order(raw: dict[str, Any]) -> ExchangeOrder:
    algo_status = raw.get("algoStatus", "NEW")
    status = _ALGO_STATUS_MAP.get(algo_status, OrderStatus.NEW)
    order_type_str = raw.get("orderType", raw.get("type", "STOP_MARKET"))
    return ExchangeOrder(
        order_id=int(raw.get("actualOrderId", 0) or 0),
        symbol=str(raw["symbol"]),
        side=OrderSide(raw["side"]),
        order_type=OrderType(order_type_str),
        quantity=float(raw.get("quantity", 0)),
        price=float(raw.get("price", 0)),
        stop_price=float(raw.get("triggerPrice", 0)),
        status=status,
        filled_qty=float(raw.get("quantity", 0)) if status is OrderStatus.FILLED else 0.0,
        avg_fill_price=float(raw.get("actualPrice", 0)),
        created_at=_from_millis(int(raw["createTime"])) if "createTime" in raw else None,
        updated_at=_from_millis(int(raw["updateTime"])) if "updateTime" in raw else None,
        algo_id=int(raw["algoId"]),
    )


_TIME_SYNC_INTERVAL = 300  # re-sync every 5 minutes


class BinanceFuturesClient:
    """Authenticated Binance Futures REST client (stdlib-only)."""

    def __init__(self, config: LiveConfig) -> None:
        self._api_key = config.api_key
        self._api_secret = config.api_secret.encode()
        self._base_url = config.base_url
        self._rate_limiter = _RateLimiter(limit_per_minute=2400)
        self._time_offset_ms: int = 0
        self._last_time_sync_monotonic: float = 0.0
        self._last_sync_wall_ms: int = 0
        self._last_sync_monotonic_ms: int = 0
        self._sync_server_time()

    # -- Time sync -------------------------------------------------------------

    def _sync_server_time(self) -> None:
        """Calculate offset between local clock and Binance server time."""
        try:
            local_before = int(time.time() * 1000)
            url = f"{self._base_url}/fapi/v1/time"
            with urlopen(url) as response:
                server_time = json.loads(response.read())["serverTime"]
            local_after = int(time.time() * 1000)
            local_mid = (local_before + local_after) // 2
            self._time_offset_ms = server_time - local_mid
            self._last_time_sync_monotonic = time.monotonic()
            self._last_sync_wall_ms = int(time.time() * 1000)
            self._last_sync_monotonic_ms = int(time.monotonic() * 1000)
            if abs(self._time_offset_ms) > 500:
                print(
                    f"Clock offset: {self._time_offset_ms:+d}ms (synced with Binance)",
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
            print("Detected local clock jump; re-syncing with Binance.", file=sys.stderr)
            self._sync_server_time()

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def server_now(self) -> datetime:
        """Return current UTC time corrected for local clock drift."""
        self._ensure_time_sync()
        ms = self._timestamp_ms()
        return datetime.fromtimestamp(ms / 1000, tz=UTC)

    # -- Signing ---------------------------------------------------------------

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_time_sync()
        signed_params = dict(params)
        signed_params["timestamp"] = self._timestamp_ms()
        signed_params["recvWindow"] = _RECV_WINDOW
        query = urlencode(signed_params)
        signature = hmac.new(self._api_secret, query.encode(), hashlib.sha256).hexdigest()
        signed_params["signature"] = signature
        return signed_params

    @staticmethod
    def _read_error_body(exc: HTTPError) -> tuple[int | None, str]:
        error_body = ""
        try:
            error_body = exc.read().decode()
        except Exception:
            return None, error_body

        try:
            payload = json.loads(error_body)
        except json.JSONDecodeError:
            return None, error_body

        code = payload.get("code")
        if isinstance(code, int):
            return code, error_body
        return None, error_body

    # -- HTTP ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        weight: int = 1,
    ) -> Any:
        if params is None:
            params = {}

        headers = {"X-MBX-APIKEY": self._api_key}

        next_delay = _DEFAULT_RETRY_DELAY
        attempt = 1
        timestamp_retry_count = 0

        while True:
            request_params = self._sign(params) if signed else dict(params)
            query = urlencode(request_params)
            if method == "GET":
                url = f"{self._base_url}{path}?{query}"
                body = None
            else:
                url = f"{self._base_url}{path}"
                body = query.encode()

            request_headers = dict(headers)
            if body is not None:
                request_headers["Content-Type"] = "application/x-www-form-urlencoded"

            self._rate_limiter.acquire(weight)
            req = Request(url, data=body, headers=request_headers, method=method)
            try:
                with urlopen(req) as response:
                    payload = json.loads(response.read())
                    _sync_rate_limiter(self._rate_limiter, response.headers)
                    return payload
            except HTTPError as exc:
                error_code, error_body = self._read_error_body(exc)
                if signed and error_code == -1021 and timestamp_retry_count < _MAX_TIMESTAMP_RETRIES:
                    timestamp_retry_count += 1
                    print(
                        f"Binance rejected timestamp on {path}; re-syncing server time and retrying.",
                        file=sys.stderr,
                    )
                    self._ensure_time_sync(force=True)
                    continue
                if exc.code not in _RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"Binance API error {exc.code} on {method} {path}: {error_body}"
                    ) from exc
                _sync_rate_limiter(self._rate_limiter, exc.headers)
                delay = _retry_delay(exc, next_delay)
                print(
                    f"Binance throttled ({exc.code}) on {path}; "
                    f"waiting {delay:.1f}s before retry {attempt + 1}.",
                    file=sys.stderr,
                )
                time.sleep(delay)
                next_delay = min(max(delay * 2, _DEFAULT_RETRY_DELAY), _MAX_RETRY_DELAY)
                attempt += 1

    # -- Order methods ---------------------------------------------------------

    def place_market_order(
        self, symbol: str, side: OrderSide, quantity: float,
        position_side: str = "BOTH",
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "symbol": _symbol_for_api(symbol),
            "side": side.value,
            "type": "MARKET",
            "quantity": quantity,
            "positionSide": position_side,
        }
        return _parse_order(self._request("POST", "/fapi/v1/order", params, weight=1))

    def place_limit_order(
        self, symbol: str, side: OrderSide, quantity: float, price: float,
        position_side: str = "BOTH",
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "symbol": _symbol_for_api(symbol),
            "side": side.value,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
            "positionSide": position_side,
        }
        return _parse_order(self._request("POST", "/fapi/v1/order", params, weight=1))

    def place_stop_market(
        self, symbol: str, side: OrderSide, stop_price: float,
        position_side: str = "BOTH", quantity: float | None = None,
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": _symbol_for_api(symbol),
            "side": side.value,
            "type": "STOP_MARKET",
            "triggerPrice": stop_price,
            "positionSide": position_side,
        }
        if quantity is not None:
            params["quantity"] = quantity
        else:
            params["closePosition"] = "true"
        return _parse_algo_order(self._request("POST", "/fapi/v1/algoOrder", params, weight=1))

    def place_take_profit_market(
        self, symbol: str, side: OrderSide, stop_price: float,
        position_side: str = "BOTH", quantity: float | None = None,
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": _symbol_for_api(symbol),
            "side": side.value,
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": stop_price,
            "positionSide": position_side,
        }
        if quantity is not None:
            params["quantity"] = quantity
        else:
            params["closePosition"] = "true"
        return _parse_algo_order(self._request("POST", "/fapi/v1/algoOrder", params, weight=1))

    def cancel_order(self, symbol: str, order_id: int) -> ExchangeOrder:
        params: dict[str, Any] = {
            "symbol": _symbol_for_api(symbol),
            "orderId": order_id,
        }
        return _parse_order(self._request("DELETE", "/fapi/v1/order", params, weight=1))

    def get_order(self, symbol: str, order_id: int) -> ExchangeOrder:
        params: dict[str, Any] = {
            "symbol": _symbol_for_api(symbol),
            "orderId": order_id,
        }
        return _parse_order(self._request("GET", "/fapi/v1/order", params, weight=1))

    def get_algo_order(self, algo_id: int) -> ExchangeOrder:
        params: dict[str, Any] = {"algoId": algo_id}
        return _parse_algo_order(self._request("GET", "/fapi/v1/algoOrder", params, weight=1))

    def cancel_algo_order(self, algo_id: int) -> dict[str, Any]:
        params: dict[str, Any] = {"algoId": algo_id}
        return self._request("DELETE", "/fapi/v1/algoOrder", params, weight=1)

    def get_open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = _symbol_for_api(symbol)
        raw = self._request("GET", "/fapi/v1/openOrders", params, weight=1)
        return [_parse_order(r) for r in raw]

    def get_position_info(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = _symbol_for_api(symbol)
        return self._request("GET", "/fapi/v2/positionRisk", params, weight=5)

    def get_account_info(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", {}, weight=5)

    def get_available_balance(self) -> float:
        """Return the available USDT balance from the futures wallet."""
        info = self.get_account_info()
        return float(info.get("availableBalance", 0))

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": _symbol_for_api(symbol),
            "leverage": leverage,
        }
        return self._request("POST", "/fapi/v1/leverage", params, weight=1)

    def get_exchange_info(self) -> dict[str, Any]:
        return self._request(
            "GET", "/fapi/v1/exchangeInfo", {}, signed=False, weight=1,
        )

    def get_mark_price(self, symbol: str) -> float:
        params: dict[str, Any] = {"symbol": _symbol_for_api(symbol)}
        raw = self._request("GET", "/fapi/v1/premiumIndex", params, signed=False, weight=1)
        return float(raw["markPrice"])


class LiveMarketClient(BinanceClient):
    """No-cache subclass of BinanceClient for live market data.

    Signal generators receive this so they always get fresh klines / trades.
    """

    def _get_json(self, path: str, params: dict[str, Any], *, weight: int) -> Any:
        # Skip both memory and disk cache — always hit the API
        query = urlencode(params)
        url = f"{self._base_url}{path}?{query}"
        next_delay = _DEFAULT_RETRY_DELAY

        attempt = 1
        while True:
            self._rate_limiter.acquire(weight)
            try:
                with urlopen(url) as response:
                    payload = json.loads(response.read())
                    _sync_rate_limiter(self._rate_limiter, response.headers)
                    return payload
            except HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS_CODES:
                    raise
                _sync_rate_limiter(self._rate_limiter, exc.headers)
                delay = _retry_delay(exc, next_delay)
                symbol = params.get("symbol", "")
                print(
                    f"Binance throttled ({exc.code}) on {path} symbol={symbol}; "
                    f"waiting {delay:.1f}s before retry {attempt + 1}.",
                    file=sys.stderr,
                )
                time.sleep(delay)
                next_delay = min(max(delay * 2, _DEFAULT_RETRY_DELAY), _MAX_RETRY_DELAY)
                attempt += 1
