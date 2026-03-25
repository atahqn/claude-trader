from __future__ import annotations

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


def build_market_data_bundle(
    client: "BinanceClient",
    symbols: list[str],
    start: datetime,
    end: datetime,
    request: MarketDataRequest,
) -> MarketDataBundle:
    bundle = MarketDataBundle(request=request, start=start, end=end)

    for symbol in symbols:
        symbol_data = SymbolMarketData(symbol=symbol)
        for requirement in request.datasets:
            symbol_data.set(
                requirement,
                _fetch_dataset(
                    client,
                    requirement,
                    symbol,
                    start,
                    end,
                    request,
                ),
            )
        bundle.by_symbol[symbol] = symbol_data

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
