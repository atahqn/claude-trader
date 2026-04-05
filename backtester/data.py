from __future__ import annotations

import hashlib
import json
import sys
import threading
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
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        tmp.replace(path)

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
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.limit_per_minute)
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, weight: int) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= weight:
                    self._tokens -= weight
                    return
            time.sleep(0.1)

    def sync_from_server(self, used_weight: int) -> None:
        with self._lock:
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


def _parse_kline(row: list[Any]) -> Candle:
    return Candle(
        open_time=_from_millis(row[0]),
        close_time=_from_millis(row[6]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        taker_buy_volume=float(row[9]) if len(row) > 9 else 0.0,
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
        timestamp=_from_millis(int(row["fundingTime"])),
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
# BinanceClient
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BinanceClient:
    market_type: MarketType = MarketType.FUTURES
    cache_root: Path = _CACHE_ROOT
    _disk_cache: _DiskCache = field(init=False, repr=False)
    _mem_cache: dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    _mem_lock: threading.Lock = field(init=False, repr=False)
    _rate_limiter: _RateLimiter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._disk_cache = _DiskCache(self.cache_root)
        self._mem_lock = threading.Lock()
        limit = 2400 if self.market_type is MarketType.FUTURES else 6000
        self._rate_limiter = _RateLimiter(limit_per_minute=limit)

    @property
    def _base_url(self) -> str:
        if self.market_type is MarketType.FUTURES:
            return "https://fapi.binance.com"
        return "https://api.binance.com"

    @property
    def _klines_path(self) -> str:
        if self.market_type is MarketType.FUTURES:
            return "/fapi/v1/klines"
        return "/api/v3/klines"

    @property
    def _agg_trades_path(self) -> str:
        if self.market_type is MarketType.FUTURES:
            return "/fapi/v1/aggTrades"
        return "/api/v3/aggTrades"

    # -- Public API ----------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Fetch klines with automatic pagination (batch of 1500)."""
        api_symbol = _symbol_for_api(symbol)
        batch_limit = 1500 if self.market_type is MarketType.FUTURES else 1000
        all_candles: list[Candle] = []
        cursor_ms = _to_millis(start)
        end_ms = _to_millis(end)

        while cursor_ms < end_ms:
            params: dict[str, Any] = {
                "symbol": api_symbol,
                "interval": interval,
                "limit": batch_limit,
                "startTime": cursor_ms,
                "endTime": end_ms,
            }
            payload = self._get_json(self._klines_path, params, weight=2)
            if not payload:
                break
            batch = [_parse_kline(row) for row in payload]
            all_candles.extend(batch)
            if len(batch) < batch_limit:
                break
            # Advance cursor past last candle
            cursor_ms = _to_millis(batch[-1].open_time) + 1

        return all_candles

    def fetch_agg_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[AggTrade]:
        """Fetch aggregate trades with fromId pagination and dedup."""
        api_symbol = _symbol_for_api(symbol)
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)
        all_trades: list[AggTrade] = []
        seen_ids: set[int] = set()

        # First request uses startTime/endTime
        params: dict[str, Any] = {
            "symbol": api_symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        payload = self._get_json(self._agg_trades_path, params, weight=20)
        if not payload:
            return all_trades

        while True:
            next_from_id: int | None = None
            stop = False
            for raw in payload:
                trade = _parse_agg_trade(raw)
                if trade.trade_id in seen_ids:
                    continue
                seen_ids.add(trade.trade_id)
                trade_ms = _to_millis(trade.timestamp)
                if trade_ms < start_ms:
                    next_from_id = trade.trade_id + 1
                    continue
                if trade_ms >= end_ms:
                    stop = True
                    break
                all_trades.append(trade)
                next_from_id = trade.trade_id + 1

            if stop or len(payload) < 1000 or next_from_id is None:
                break

            params = {
                "symbol": api_symbol,
                "fromId": next_from_id,
                "limit": 1000,
            }
            payload = self._get_json(self._agg_trades_path, params, weight=20)
            if not payload:
                break

        return all_trades

    def fetch_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[FundingRate]:
        self._require_futures_dataset("funding rates")
        return self._fetch_timeseries(
            path="/fapi/v1/fundingRate",
            params={"symbol": _symbol_for_api(symbol)},
            start=start,
            end=end,
            limit=1000,
            time_field="fundingTime",
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
            path="/fapi/v1/markPriceKlines",
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
            path="/fapi/v1/premiumIndexKlines",
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
        max_workers: int = 8,
    ) -> MarketDataBundle:
        from marketdata.bundle import build_market_data_bundle

        return build_market_data_bundle(self, symbols, start, end, request, max_workers=max_workers)

    def fetch_market_context_bundle(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        request: MarketDataRequest,
        max_workers: int = 8,
    ):
        from marketdata.context import fetch_market_context_bundle

        return fetch_market_context_bundle(self, symbols, start, end, request, max_workers=max_workers)

    # -- Internal ------------------------------------------------------------

    def _require_futures_dataset(self, label: str) -> None:
        if self.market_type is not MarketType.FUTURES:
            raise ValueError(f"{label} are only available for Binance futures")

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
        batch_limit = 1500
        all_candles: list[Candle] = []
        cursor_ms = _to_millis(start)
        end_ms = _to_millis(end)

        while cursor_ms < end_ms:
            params: dict[str, Any] = {
                "symbol": api_symbol,
                "interval": interval,
                "limit": batch_limit,
                "startTime": cursor_ms,
                "endTime": end_ms,
            }
            payload = self._get_json(path, params, weight=weight)
            if not payload:
                break
            batch = [_parse_kline(row) for row in payload]
            all_candles.extend(batch)
            if len(batch) < batch_limit:
                break
            cursor_ms = _to_millis(batch[-1].open_time) + 1

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
            if not payload:
                break

            last_ms: int | None = None
            for raw in payload:
                raw_ms = int(raw[time_field])
                if raw_ms < cursor_ms:
                    continue
                if raw_ms >= end_ms:
                    continue
                rows.append(parser(raw))
                last_ms = raw_ms

            if len(payload) < limit or last_ms is None:
                break
            cursor_ms = last_ms + 1

        return rows

    def _get_json(self, path: str, params: dict[str, Any], *, weight: int) -> Any:
        # In-memory cache
        mem_key = path + "|" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        with self._mem_lock:
            if mem_key in self._mem_cache:
                return self._mem_cache[mem_key]

        # Disk cache
        cache_key = [path] + [f"{k}={params[k]}" for k in sorted(params)]
        cached = self._disk_cache.get("binance", cache_key)
        if cached is not None:
            with self._mem_lock:
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
                    _sync_rate_limiter(self._rate_limiter, response.headers)
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
                continue

            self._disk_cache.set("binance", cache_key, payload)
            with self._mem_lock:
                self._mem_cache[mem_key] = payload
            return payload
