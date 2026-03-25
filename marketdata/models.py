from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class DataRequirement(StrEnum):
    OHLCV = "ohlcv"
    AGG_TRADES = "agg_trades"
    FUNDING_RATES = "funding_rates"
    MARK_PRICE_KLINES = "mark_price_klines"
    PREMIUM_INDEX_KLINES = "premium_index_klines"


@dataclass(slots=True, frozen=True)
class FundingRate:
    timestamp: datetime
    funding_rate: float
    mark_price: float | None = None


@dataclass(slots=True, frozen=True)
class MarketDataRequest:
    datasets: frozenset[DataRequirement] = field(
        default_factory=lambda: frozenset({DataRequirement.OHLCV})
    )
    ohlcv_interval: str = "1h"

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("at least one dataset is required")

    @classmethod
    def ohlcv_only(cls, interval: str = "1h") -> "MarketDataRequest":
        return cls(
            datasets=frozenset({DataRequirement.OHLCV}),
            ohlcv_interval=interval,
        )


@dataclass(slots=True)
class SymbolMarketData:
    symbol: str
    datasets: dict[DataRequirement, list[Any]] = field(default_factory=dict)

    def get(self, requirement: DataRequirement) -> list[Any]:
        return self.datasets.get(requirement, [])

    def set(self, requirement: DataRequirement, rows: list[Any]) -> None:
        self.datasets[requirement] = rows

    def has(self, requirement: DataRequirement) -> bool:
        return requirement in self.datasets


@dataclass(slots=True)
class MarketDataBundle:
    request: MarketDataRequest
    start: datetime
    end: datetime
    by_symbol: dict[str, SymbolMarketData] = field(default_factory=dict)

    def for_symbol(self, symbol: str) -> SymbolMarketData:
        return self.by_symbol[symbol]
