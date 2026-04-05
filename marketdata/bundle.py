from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TYPE_CHECKING

from .models import (
    DataRequirement,
    MarketDataBundle,
    MarketDataRequest,
    SymbolMarketData,
)

if TYPE_CHECKING:
    from backtester.data import BinanceClient


def _fetch_symbol_data(
    client: "BinanceClient",
    symbol: str,
    start: datetime,
    end: datetime,
    request: MarketDataRequest,
) -> tuple[str, SymbolMarketData]:
    symbol_data = SymbolMarketData(symbol=symbol)
    for requirement in request.datasets:
        symbol_data.set(
            requirement,
            _fetch_dataset(client, requirement, symbol, start, end, request),
        )
    return symbol, symbol_data


def build_market_data_bundle(
    client: "BinanceClient",
    symbols: list[str],
    start: datetime,
    end: datetime,
    request: MarketDataRequest,
    *,
    max_workers: int = 8,
) -> MarketDataBundle:
    bundle = MarketDataBundle(request=request, start=start, end=end)
    if not symbols:
        return bundle

    if max_workers <= 1 or len(symbols) <= 1:
        for symbol in symbols:
            _, symbol_data = _fetch_symbol_data(client, symbol, start, end, request)
            bundle.by_symbol[symbol] = symbol_data
    else:
        results: dict[str, SymbolMarketData] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as pool:
            futures = {
                pool.submit(_fetch_symbol_data, client, symbol, start, end, request): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                sym, sym_data = future.result()
                results[sym] = sym_data
        # Preserve original symbol order.
        for symbol in symbols:
            bundle.by_symbol[symbol] = results[symbol]

    return bundle


def _fetch_dataset(
    client: "BinanceClient",
    requirement: DataRequirement,
    symbol: str,
    start: datetime,
    end: datetime,
    request: MarketDataRequest,
) -> list:
    if requirement is DataRequirement.OHLCV:
        return client.fetch_klines(symbol, request.ohlcv_interval, start, end)
    if requirement is DataRequirement.AGG_TRADES:
        return client.fetch_agg_trades(symbol, start, end)
    if requirement is DataRequirement.FUNDING_RATES:
        return client.fetch_funding_rates(symbol, start, end)
    if requirement is DataRequirement.MARK_PRICE_KLINES:
        return client.fetch_mark_price_klines(symbol, request.ohlcv_interval, start, end)
    if requirement is DataRequirement.PREMIUM_INDEX_KLINES:
        return client.fetch_premium_index_klines(symbol, request.ohlcv_interval, start, end)
    raise ValueError(f"unsupported data requirement: {requirement}")
