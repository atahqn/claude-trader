from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from marketdata.models import (
    FundingRate,
    MarketDataBundle,
    MarketDataRequest,
)

from .models import AggTrade, Candle, MarketType

_DEFAULT_RETRY_DELAY = 15.0
_MAX_RETRY_DELAY = 300.0
_RETRYABLE_STATUS_CODES = {418, 429}

_CACHE_ROOT = Path.home() / ".claude_trader" / "cache"
_BYBIT_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}


# ---------------------------------------------------------------------------
# Disk cache (SHA-256 keyed JSON, ported from kriptistan cache.py)
# ---------------------------------------------------------------------------

class _DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, namespace: str, key_parts: list[str]) -> Any | None:
        path = self._path(namespace, key_parts)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, namespace: str, key_parts: list[str], payload: Any) -> None:
        path = self._path(namespace, key_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))

    def _path(self, namespace: str, key_parts: list[str]) -> Path:
        digest = hashlib.sha256("::".join(key_parts).encode()).hexdigest()
        return self.root / namespace / f"{digest}.json"


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket, ported from kriptistan data_binance.py)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _RateLimiter:
    limit_per_minute: int = 2400
    _tokens: float = field(init=False, repr=False)
    _updated_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.limit_per_minute)
        self._updated_at = time.monotonic()

    def acquire(self, weight: int) -> None:
        while True:
            self._refill()
            if self._tokens >= weight:
                self._tokens -= weight
                return
            time.sleep(0.1)

    def sync_from_server(self, used_weight: int) -> None:
        self._refill()
        server_remaining = max(0.0, float(self.limit_per_minute) - used_weight)
        if server_remaining < self._tokens:
            self._tokens = server_remaining

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        refill = (self.limit_per_minute / 60) * elapsed
        self._tokens = min(float(self.limit_per_minute), self._tokens + refill)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_millis(dt: datetime) -> int:
    normalized = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    return int(normalized.timestamp() * 1000)


def _from_millis(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _interval_to_bybit(interval: str) -> str:
    mapped = _BYBIT_INTERVAL_MAP.get(interval)
    if mapped is None:
        raise ValueError(f"unsupported interval for Bybit: {interval}")
    return mapped


def _interval_to_timedelta(interval: str) -> timedelta:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"unsupported interval for Bybit: {interval}")


def _parse_kline(row: list[Any], interval: str) -> Candle:
    open_time = _from_millis(int(row[0]))
    close_time = open_time + _interval_to_timedelta(interval)
    return Candle(
        open_time=open_time,
        close_time=close_time,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]) if len(row) > 5 else 0.0,
    )


def _parse_agg_trade(row: dict[str, Any]) -> AggTrade:
    return AggTrade(
        trade_id=int(row["a"]),
        timestamp=_from_millis(row["T"]),
        price=float(row["p"]),
        quantity=float(row["q"]),
    )


def _parse_funding_rate(row: dict[str, Any]) -> FundingRate:
    mark_price_raw = row.get("markPrice")
    return FundingRate(
        timestamp=_from_millis(int(row.get("fundingRateTimestamp", row.get("fundingTime")))),
        funding_rate=float(row["fundingRate"]),
        mark_price=float(mark_price_raw) if mark_price_raw not in (None, "") else None,
    )


def _symbol_for_api(ticker: str) -> str:
    """Convert 'BTC/USDT' -> 'BTCUSDT'."""
    return ticker.replace("/", "")


def _sync_rate_limiter(rate_limiter: _RateLimiter, headers: Any) -> None:
    if headers is None:
        return
    used = headers.get("X-MBX-USED-WEIGHT-1m")
    if used is None:
        limit_status = headers.get("X-Bapi-Limit-Status")
        limit = headers.get("X-Bapi-Limit")
        if limit_status is not None and limit is not None:
            try:
                rate_limiter.limit_per_minute = int(limit) * 60
                rate_limiter.sync_from_server(int(limit) - int(limit_status))
                return
            except (ValueError, TypeError):
                pass
    if used is not None:
        try:
            rate_limiter.sync_from_server(int(used))
        except (ValueError, TypeError):
            pass


def _retry_delay(exc: HTTPError, fallback: float) -> float:
    if exc.headers is not None:
        retry_after = exc.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
    return fallback


# ---------------------------------------------------------------------------
# BybitClient
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BybitClient:
    market_type: MarketType = MarketType.FUTURES
    cache_root: Path = _CACHE_ROOT
    base_url: str = "https://api.bybit.com"
    _disk_cache: _DiskCache = field(init=False, repr=False)
    _mem_cache: dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    _rate_limiter: _RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._disk_cache = _DiskCache(self.cache_root)
        limit = 1200
        self._rate_limiter = _RateLimiter(limit_per_minute=limit)

    @property
    def _base_url(self) -> str:
        return self.base_url

    @property
    def _category(self) -> str:
        if self.market_type is MarketType.FUTURES:
            return "linear"
        return "spot"

    # -- Public API ----------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Fetch Bybit klines with automatic pagination."""
        api_symbol = _symbol_for_api(symbol)
        bybit_interval = _interval_to_bybit(interval)
        batch_limit = 1000
        all_candles: list[Candle] = []
        cursor_ms = _to_millis(start)
        end_ms = _to_millis(end)

        while cursor_ms < end_ms:
            params: dict[str, Any] = {
                "category": self._category,
                "symbol": api_symbol,
                "interval": bybit_interval,
                "limit": batch_limit,
                "start": cursor_ms,
                "end": end_ms,
            }
            payload = self._get_json("/v5/market/kline", params, weight=1)
            raw_rows = payload.get("list", [])
            if not raw_rows:
                break
            batch = sorted(
                (
                    _parse_kline(row, interval)
                    for row in raw_rows
                ),
                key=lambda candle: candle.open_time,
            )
            batch = [
                candle
                for candle in batch
                if candle.open_time >= start and candle.open_time < end
            ]
            if not batch:
                break
            all_candles.extend(batch)
            if len(raw_rows) < batch_limit:
                break
            cursor_ms = _to_millis(batch[-1].open_time + _interval_to_timedelta(interval))

        return all_candles

    def fetch_agg_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[AggTrade]:
        raise NotImplementedError(
            "Bybit V5 does not expose time-paginated historical public trades "
            "compatible with the old aggTrades workflow. "
            "Use approximate=True for Bybit backtests."
        )

    def fetch_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[FundingRate]:
        self._require_futures_dataset("funding rates")
        return self._fetch_timeseries(
            path="/v5/market/funding/history",
            params={
                "category": self._category,
                "symbol": _symbol_for_api(symbol),
            },
            start=start,
            end=end,
            limit=200,
            time_field="fundingRateTimestamp",
            weight=1,
            parser=_parse_funding_rate,
        )

    def fetch_mark_price_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        self._require_futures_dataset("mark price klines")
        return self._fetch_kline_series(
            path="/v5/market/mark-price-kline",
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            weight=1,
        )

    def fetch_premium_index_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        self._require_futures_dataset("premium index klines")
        return self._fetch_kline_series(
            path="/v5/market/premium-index-price-kline",
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            weight=1,
        )

    def fetch_market_data_bundle(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        request: MarketDataRequest,
    ) -> MarketDataBundle:
        from marketdata.bundle import build_market_data_bundle

        return build_market_data_bundle(self, symbols, start, end, request)

    def fetch_market_context_bundle(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        request: MarketDataRequest,
    ):
        from marketdata.context import fetch_market_context_bundle

        return fetch_market_context_bundle(self, symbols, start, end, request)

    # -- Internal ------------------------------------------------------------

    def _require_futures_dataset(self, label: str) -> None:
        if self.market_type is not MarketType.FUTURES:
            raise ValueError(f"{label} are only available for Bybit linear futures")

    def _fetch_kline_series(
        self,
        *,
        path: str,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        weight: int,
    ) -> list[Candle]:
        api_symbol = _symbol_for_api(symbol)
        bybit_interval = _interval_to_bybit(interval)
        batch_limit = 1000
        all_candles: list[Candle] = []
        cursor_ms = _to_millis(start)
        end_ms = _to_millis(end)

        while cursor_ms < end_ms:
            params: dict[str, Any] = {
                "category": self._category,
                "symbol": api_symbol,
                "interval": bybit_interval,
                "limit": batch_limit,
                "start": cursor_ms,
                "end": end_ms,
            }
            payload = self._get_json(path, params, weight=weight)
            raw_rows = payload.get("list", [])
            if not raw_rows:
                break
            batch = sorted(
                (_parse_kline(row, interval) for row in raw_rows),
                key=lambda candle: candle.open_time,
            )
            batch = [
                candle
                for candle in batch
                if candle.open_time >= start and candle.open_time < end
            ]
            if not batch:
                break
            all_candles.extend(batch)
            if len(raw_rows) < batch_limit:
                break
            cursor_ms = _to_millis(batch[-1].open_time + _interval_to_timedelta(interval))

        return all_candles

    def _fetch_timeseries(
        self,
        *,
        path: str,
        params: dict[str, Any],
        start: datetime,
        end: datetime,
        limit: int,
        time_field: str,
        weight: int,
        parser: Any,
    ) -> list[Any]:
        rows: list[Any] = []
        cursor_ms = _to_millis(start)
        end_ms = _to_millis(end)

        while cursor_ms < end_ms:
            request_params = dict(params)
            request_params["startTime"] = cursor_ms
            request_params["endTime"] = end_ms
            request_params["limit"] = limit
            payload = self._get_json(path, request_params, weight=weight)
            raw_rows = payload.get("list", [])
            if not raw_rows:
                break

            last_ms: int | None = None
            for raw in sorted(raw_rows, key=lambda item: int(item[time_field])):
                raw_ms = int(raw[time_field])
                if raw_ms < cursor_ms:
                    continue
                if raw_ms >= end_ms:
                    continue
                rows.append(parser(raw))
                last_ms = raw_ms

            if len(raw_rows) < limit or last_ms is None:
                break
            cursor_ms = last_ms + 1

        return rows

    def _get_json(self, path: str, params: dict[str, Any], *, weight: int) -> Any:
        # In-memory cache
        mem_key = path + "|" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        if mem_key in self._mem_cache:
            return self._mem_cache[mem_key]

        # Disk cache
        cache_key = [path] + [f"{k}={params[k]}" for k in sorted(params)]
        cached = self._disk_cache.get("bybit", cache_key)
        if cached is not None:
            self._mem_cache[mem_key] = cached
            return cached

        # HTTP request with retry
        query = urlencode(params)
        url = f"{self._base_url}{path}?{query}"
        next_delay = _DEFAULT_RETRY_DELAY
        attempt = 1

        while True:
            self._rate_limiter.acquire(weight)
            try:
                with urlopen(url) as response:
                    payload = json.loads(response.read())
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
                continue

            if payload.get("retCode", 0) != 0:
                raise RuntimeError(
                    f"Bybit API error {payload.get('retCode')} on {path}: "
                    f"{payload.get('retMsg', 'unknown error')}"
                )

            result = payload.get("result", payload)
            self._disk_cache.set("bybit", cache_key, result)
            self._mem_cache[mem_key] = result
            return result
