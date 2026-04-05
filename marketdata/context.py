from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtester.models import Candle

from .models import (
    DataRequirement,
    FundingRate,
    MarketDataBundle,
    MarketDataRequest,
    SymbolMarketData,
)

if TYPE_CHECKING:
    from backtester.data import BinanceClient


@dataclass(slots=True)
class SymbolMarketContext:
    symbol: str
    frame: pd.DataFrame
    request: MarketDataRequest
    raw_datasets: dict[DataRequirement, list[Any]] = field(default_factory=dict)

    def latest_row(self) -> pd.Series:
        return self.frame.iloc[-1]

    def previous_row(self) -> pd.Series:
        return self.frame.iloc[-2]

    def raw(self, requirement: DataRequirement) -> list[Any]:
        return self.raw_datasets.get(requirement, [])


@dataclass(slots=True)
class MarketContextBundle:
    request: MarketDataRequest
    start: datetime
    end: datetime
    by_symbol: dict[str, SymbolMarketContext] = field(default_factory=dict)

    def for_symbol(self, symbol: str) -> SymbolMarketContext:
        return self.by_symbol[symbol]


def build_market_context_bundle(raw_bundle: MarketDataBundle) -> MarketContextBundle:
    if DataRequirement.OHLCV not in raw_bundle.request.datasets:
        raise ValueError("market context requires OHLCV in the request")

    context_bundle = MarketContextBundle(
        request=raw_bundle.request,
        start=raw_bundle.start,
        end=raw_bundle.end,
    )

    for symbol, symbol_data in raw_bundle.by_symbol.items():
        context_bundle.by_symbol[symbol] = build_symbol_market_context(
            symbol_data,
            raw_bundle.request,
        )

    return context_bundle


def fetch_market_context_bundle(
    client: "BinanceClient",
    symbols: list[str],
    start: datetime,
    end: datetime,
    request: MarketDataRequest,
    max_workers: int = 8,
) -> MarketContextBundle:
    return build_market_context_bundle(
        client.fetch_market_data_bundle(symbols, start, end, request, max_workers=max_workers)
    )


def build_symbol_market_context(
    symbol_data: SymbolMarketData,
    request: MarketDataRequest,
) -> SymbolMarketContext:
    if not symbol_data.has(DataRequirement.OHLCV):
        raise ValueError(f"{symbol_data.symbol} is missing OHLCV data")

    base = _candles_to_frame(symbol_data.get(DataRequirement.OHLCV))
    raw_passthrough: dict[DataRequirement, list[Any]] = {}

    for requirement, rows in symbol_data.datasets.items():
        if requirement is DataRequirement.OHLCV:
            continue
        if requirement is DataRequirement.AGG_TRADES:
            raw_passthrough[requirement] = rows
            continue
        aligned = _aligned_frame(requirement, rows)
        if aligned is None:
            raw_passthrough[requirement] = rows
            continue
        base = _merge_asof(base, aligned)

    return SymbolMarketContext(
        symbol=symbol_data.symbol,
        frame=base,
        request=request,
        raw_datasets=raw_passthrough,
    )


def _candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    rows = [
        {
            "open_time": candle.open_time,
            "close_time": candle.close_time,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "taker_buy_volume": candle.taker_buy_volume,
        }
        for candle in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("close_time").reset_index(drop=True)


def _aligned_frame(requirement: DataRequirement, rows: list[Any]) -> pd.DataFrame | None:
    if requirement is DataRequirement.FUNDING_RATES:
        return _funding_frame(rows)
    if requirement is DataRequirement.MARK_PRICE_KLINES:
        return _prefixed_candle_frame("mark", rows)
    if requirement is DataRequirement.PREMIUM_INDEX_KLINES:
        return _prefixed_candle_frame("premium", rows)
    return None


def _merge_asof(base: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    if base.empty or aligned.empty:
        return base
    return pd.merge_asof(
        base.sort_values("close_time"),
        aligned.sort_values("asof_time"),
        left_on="close_time",
        right_on="asof_time",
        direction="backward",
        allow_exact_matches=True,
    ).drop(columns=["asof_time"])


def _funding_frame(rows: list[FundingRate]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "asof_time": row.timestamp,
            "funding_timestamp": row.timestamp,
            "funding_rate": row.funding_rate,
            "funding_mark_price": row.mark_price,
        }
        for row in rows
    ])
    return _sorted_frame(frame)


def _prefixed_candle_frame(prefix: str, candles: list[Candle]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "asof_time": candle.close_time,
            f"{prefix}_open_time": candle.open_time,
            f"{prefix}_close_time": candle.close_time,
            f"{prefix}_open": candle.open,
            f"{prefix}_high": candle.high,
            f"{prefix}_low": candle.low,
            f"{prefix}_close": candle.close,
            f"{prefix}_volume": candle.volume,
        }
        for candle in candles
    ])
    return _sorted_frame(frame)


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values("asof_time").reset_index(drop=True)
