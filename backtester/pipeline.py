from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from marketdata import (
    DataRequirement,
    MarketContextBundle,
    MarketDataRequest,
    SymbolMarketContext,
)

from .models import AggTrade, Candle
from .preview import interval_to_timedelta

if TYPE_CHECKING:
    from .data import BinanceClient


def _normalize_time(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _floor_time(dt: datetime, chunk: timedelta) -> datetime:
    normalized = _normalize_time(dt)
    chunk_seconds = int(chunk.total_seconds())
    if chunk_seconds <= 0:
        raise ValueError("chunk size must be positive")
    timestamp = int(normalized.timestamp())
    floored = timestamp - (timestamp % chunk_seconds)
    return datetime.fromtimestamp(floored, tz=UTC)


def _frame_to_candles(frame: Any) -> list[Candle]:
    if frame.empty:
        return []
    candles: list[Candle] = []
    for row in frame.itertuples(index=False):
        candles.append(
            Candle(
                open_time=row.open_time,
                close_time=row.close_time,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0)),
            )
        )
    return candles


@dataclass(slots=True)
class PreparedMarketContext:
    start: datetime
    end: datetime
    fetch_start: datetime
    request: MarketDataRequest
    bundle: MarketContextBundle
    poll_candles: dict[str, list[Candle]] = field(default_factory=dict)
    _analysis_candles: dict[str, list[Candle]] = field(init=False, repr=False, default_factory=dict)
    _poll_candles: dict[str, list[Candle]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._analysis_candles = {
            symbol: _frame_to_candles(context.frame)
            for symbol, context in self.bundle.by_symbol.items()
        }
        self._poll_candles = {
            symbol: list(rows)
            for symbol, rows in self.poll_candles.items()
        }

    @property
    def symbols(self) -> list[str]:
        return list(self.bundle.by_symbol)

    def for_symbol(self, symbol: str) -> SymbolMarketContext:
        return self.bundle.for_symbol(symbol)

    def slice_analysis_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        candles = self._analysis_candles.get(symbol, [])
        if not candles:
            return []
        lo = bisect_left(candles, start, key=lambda c: c.open_time)
        hi = bisect_left(candles, end, key=lambda c: c.open_time)
        return candles[lo:hi]

    def slice_poll_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        candles = self._poll_candles.get(symbol)
        if candles is None:
            candles = self._analysis_candles.get(symbol, [])
        if not candles:
            return []
        lo = bisect_left(candles, start, key=lambda c: c.open_time)
        hi = bisect_left(candles, end, key=lambda c: c.open_time)
        return candles[lo:hi]

    def truncated_to(self, t: datetime) -> PreparedMarketContext:
        """Return a copy with all data after *t* physically removed.

        ``start`` and ``end`` metadata are preserved unchanged.  The
        caller controls the signal generation window via explicit
        ``start``/``end`` args to ``generate_backtest_signals()``, not
        via this context's metadata.

        Used for look-ahead bias validation only.
        """
        truncated_by_symbol: dict[str, SymbolMarketContext] = {}
        for sym, smc in self.bundle.by_symbol.items():
            trunc_frame = smc.frame[smc.frame["close_time"] <= t].copy()
            trunc_raw = _truncate_raw_datasets(smc.raw_datasets, t)
            truncated_by_symbol[sym] = SymbolMarketContext(
                symbol=smc.symbol,
                frame=trunc_frame,
                request=smc.request,
                raw_datasets=trunc_raw,
            )
        truncated_bundle = MarketContextBundle(
            request=self.bundle.request,
            start=self.bundle.start,
            end=self.bundle.end,
            by_symbol=truncated_by_symbol,
        )
        truncated_poll = {
            sym: [c for c in candles if c.close_time <= t]
            for sym, candles in self.poll_candles.items()
        }
        return PreparedMarketContext(
            start=self.start,
            end=self.end,
            fetch_start=self.fetch_start,
            request=self.request,
            bundle=truncated_bundle,
            poll_candles=truncated_poll,
        )


def _truncate_raw_datasets(
    raw: dict[DataRequirement, list[Any]],
    t: datetime,
) -> dict[DataRequirement, list[Any]]:
    """Truncate every raw dataset to entries at or before *t*.

    Raises on unknown dataset types to prevent silent false negatives
    in look-ahead validation.
    """
    truncated: dict[DataRequirement, list[Any]] = {}
    for req, items in raw.items():
        if req == DataRequirement.AGG_TRADES:
            truncated[req] = [i for i in items if i.timestamp <= t]
        elif req == DataRequirement.FUNDING_RATES:
            truncated[req] = [i for i in items if i.timestamp <= t]
        elif req in (
            DataRequirement.MARK_PRICE_KLINES,
            DataRequirement.PREMIUM_INDEX_KLINES,
            DataRequirement.OHLCV,
        ):
            truncated[req] = [i for i in items if i.close_time <= t]
        else:
            raise ValueError(
                f"_truncate_raw_datasets: unknown dataset type {req!r}. "
                f"Add truncation logic before using this in validation."
            )
    return truncated


def prepare_market_context(
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    client: "BinanceClient | None" = None,
    request: MarketDataRequest | None = None,
    warmup: timedelta | None = None,
    warmup_bars: int = 0,
    max_workers: int = 8,
) -> PreparedMarketContext:
    if request is None:
        request = MarketDataRequest.ohlcv_only()
    if warmup is None:
        warmup = interval_to_timedelta(request.ohlcv_interval) * warmup_bars
    if warmup < timedelta(0):
        raise ValueError("warmup must be non-negative")

    if client is None:
        from .data import BinanceClient

        client = BinanceClient()

    fetch_start = start - warmup
    bundle = client.fetch_market_context_bundle(
        symbols, fetch_start, end, request, max_workers=max_workers,
    )
    poll_candles: dict[str, list[Candle]] = {}
    poll_interval = request.effective_poll_ohlcv_interval
    if poll_interval != request.ohlcv_interval and symbols:
        if max_workers <= 1 or len(symbols) <= 1:
            for symbol in symbols:
                poll_candles[symbol] = client.fetch_klines(
                    symbol, poll_interval, fetch_start, end,
                )
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as pool:
                futures = {
                    pool.submit(
                        client.fetch_klines, symbol, poll_interval, fetch_start, end,
                    ): symbol
                    for symbol in symbols
                }
                for future in as_completed(futures):
                    poll_candles[futures[future]] = future.result()
    return PreparedMarketContext(
        start=start,
        end=end,
        fetch_start=fetch_start,
        request=request,
        bundle=bundle,
        poll_candles=poll_candles,
    )


T = TypeVar("T")


@dataclass(slots=True)
class _ChunkedWindowCache(Generic[T]):
    chunk_size: timedelta
    fetcher: Callable[[str, datetime, datetime], list[T]]
    time_selector: Callable[[T], datetime]
    _cache: dict[tuple[str, datetime], list[T]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _in_flight: dict[tuple[str, datetime], threading.Event | BaseException] = field(
        default_factory=dict,
    )

    def _fetch_chunk(self, key: tuple[str, datetime]) -> list[T]:
        """Fetch a single chunk, coalescing concurrent requests for the same key.

        Uses an in-flight map so different keys can be fetched concurrently
        while duplicate requests for the same key are coalesced.  On failure
        the exception is stored and propagated to all waiters.
        """
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            slot = self._in_flight.get(key)
            if slot is not None:
                if isinstance(slot, BaseException):
                    raise slot
                event = slot
            else:
                event = threading.Event()
                self._in_flight[key] = event

        if slot is not None:
            # Another thread is fetching — wait for it.
            event.wait()
            with self._lock:
                if key in self._cache:
                    return self._cache[key]
                # Owner failed — stored exception is still in _in_flight.
                exc = self._in_flight.get(key)
                if isinstance(exc, BaseException):
                    raise exc
                return []  # pragma: no cover

        # We are the owner — fetch without holding the lock.
        symbol, cursor = key
        chunk_end = cursor + self.chunk_size
        try:
            data = self.fetcher(symbol, cursor, chunk_end)
        except BaseException as exc:
            with self._lock:
                self._in_flight[key] = exc
                event.set()
            raise
        with self._lock:
            self._cache[key] = data
            del self._in_flight[key]
            event.set()
        return data

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[T]:
        if end <= start:
            return []

        rows: list[T] = []
        cursor = _floor_time(start, self.chunk_size)
        while cursor < end:
            chunk = self._fetch_chunk((symbol, cursor))
            if chunk:
                lo = bisect_left(chunk, start, key=self.time_selector)
                hi = bisect_left(chunk, end, key=self.time_selector)
                rows.extend(chunk[lo:hi])
            cursor += self.chunk_size

        rows.sort(key=self.time_selector)
        return rows


@dataclass(slots=True)
class BacktestExecutionSession:
    client: Any
    prepared_context: PreparedMarketContext | None = None
    analysis_interval: str = "1h"
    minute_chunk: timedelta = timedelta(hours=6)
    agg_trade_chunk: timedelta = timedelta(minutes=15)
    use_chunk_cache: bool = True
    _minute_cache: _ChunkedWindowCache[Candle] = field(init=False, repr=False)
    _agg_trade_cache: _ChunkedWindowCache[AggTrade] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prepared_context is not None:
            self.analysis_interval = self.prepared_context.request.ohlcv_interval
        self._minute_cache = _ChunkedWindowCache(
            chunk_size=self.minute_chunk,
            fetcher=lambda symbol, start, end: self.client.fetch_klines(symbol, "1m", start, end),
            time_selector=lambda candle: candle.open_time,
        )
        self._agg_trade_cache = _ChunkedWindowCache(
            chunk_size=self.agg_trade_chunk,
            fetcher=self.client.fetch_agg_trades,
            time_selector=lambda trade: trade.timestamp,
        )

    def fetch_analysis_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        if self.prepared_context is not None and symbol in self.prepared_context.bundle.by_symbol:
            return self.prepared_context.slice_analysis_candles(symbol, start, end)
        return self.client.fetch_klines(symbol, self.analysis_interval, start, end)

    def fetch_minute_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        if not self.use_chunk_cache:
            return self.client.fetch_klines(symbol, "1m", start, end)
        return self._minute_cache.fetch(symbol, start, end)

    def fetch_agg_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[AggTrade]:
        if not self.use_chunk_cache:
            return self.client.fetch_agg_trades(symbol, start, end)
        return self._agg_trade_cache.fetch(symbol, start, end)
