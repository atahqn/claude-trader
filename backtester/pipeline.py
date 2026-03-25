from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from marketdata import MarketDataRequest, MarketContextBundle, SymbolMarketContext

from .models import AggTrade, Candle, Signal

if TYPE_CHECKING:
    from .data import BinanceClient


def _interval_to_timedelta(interval: str) -> timedelta:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise ValueError(f"unsupported interval format: {interval}")


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
    _hourly_candles: dict[str, list[Candle]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._hourly_candles = {
            symbol: _frame_to_candles(context.frame)
            for symbol, context in self.bundle.by_symbol.items()
        }

    @property
    def symbols(self) -> list[str]:
        return list(self.bundle.by_symbol)

    def for_symbol(self, symbol: str) -> SymbolMarketContext:
        return self.bundle.for_symbol(symbol)

    def slice_hourly_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        candles = self._hourly_candles.get(symbol, [])
        if not candles:
            return []
        return [
            candle
            for candle in candles
            if candle.open_time >= start and candle.open_time < end
        ]


def prepare_market_context(
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    client: "BinanceClient | None" = None,
    request: MarketDataRequest | None = None,
    warmup: timedelta | None = None,
    warmup_bars: int = 0,
) -> PreparedMarketContext:
    if request is None:
        request = MarketDataRequest.ohlcv_only()
    if warmup is None:
        warmup = _interval_to_timedelta(request.ohlcv_interval) * warmup_bars
    if warmup < timedelta(0):
        raise ValueError("warmup must be non-negative")

    if client is None:
        from .data import BinanceClient

        client = BinanceClient()

    fetch_start = start - warmup
    bundle = client.fetch_market_context_bundle(symbols, fetch_start, end, request)
    return PreparedMarketContext(
        start=start,
        end=end,
        fetch_start=fetch_start,
        request=request,
        bundle=bundle,
    )


def generate_signals_from_prepared_context(
    prepared_context: PreparedMarketContext,
    generator: Callable[[SymbolMarketContext, datetime, datetime], list[Signal]],
    *,
    symbols: list[str] | None = None,
) -> list[Signal]:
    generated: list[Signal] = []
    selected_symbols = symbols or prepared_context.symbols
    for symbol in selected_symbols:
        generated.extend(
            generator(
                prepared_context.for_symbol(symbol),
                prepared_context.start,
                prepared_context.end,
            )
        )
    return sorted(generated, key=lambda signal: signal.signal_date)


T = TypeVar("T")


@dataclass(slots=True)
class _ChunkedWindowCache(Generic[T]):
    chunk_size: timedelta
    fetcher: Callable[[str, datetime, datetime], list[T]]
    time_selector: Callable[[T], datetime]
    _cache: dict[tuple[str, datetime], list[T]] = field(default_factory=dict)

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[T]:
        if end <= start:
            return []

        rows: list[T] = []
        cursor = _floor_time(start, self.chunk_size)
        while cursor < end:
            key = (symbol, cursor)
            if key not in self._cache:
                chunk_end = cursor + self.chunk_size
                self._cache[key] = self.fetcher(symbol, cursor, chunk_end)
            rows.extend(
                row
                for row in self._cache[key]
                if start <= self.time_selector(row) < end
            )
            cursor += self.chunk_size

        rows.sort(key=self.time_selector)
        return rows


@dataclass(slots=True)
class BacktestExecutionSession:
    client: Any
    prepared_context: PreparedMarketContext | None = None
    minute_chunk: timedelta = timedelta(hours=6)
    agg_trade_chunk: timedelta = timedelta(minutes=15)
    use_chunk_cache: bool = True
    _minute_cache: _ChunkedWindowCache[Candle] = field(init=False, repr=False)
    _agg_trade_cache: _ChunkedWindowCache[AggTrade] = field(init=False, repr=False)

    def __post_init__(self) -> None:
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

    def fetch_hourly_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        if self.prepared_context is not None and symbol in self.prepared_context.bundle.by_symbol:
            return self.prepared_context.slice_hourly_candles(symbol, start, end)
        return self.client.fetch_klines(symbol, "1h", start, end)

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
