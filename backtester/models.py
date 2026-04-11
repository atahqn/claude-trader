from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class PositionType(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class MarketType(StrEnum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"


class ExitReason(StrEnum):
    TP = "TP"
    SL = "SL"
    TIMEOUT = "TIMEOUT"
    UNFILLED = "UNFILLED"


class ResolutionLevel(StrEnum):
    HOUR = "1h"
    MINUTE = "1m"
    TRADE = "trade"


@dataclass(slots=True, frozen=True)
class Signal:
    signal_date: datetime
    position_type: PositionType
    ticker: str

    tp_pct: float | None = None
    sl_pct: float | None = None
    tp_price: float | None = None
    sl_price: float | None = None

    leverage: float = 1.0
    market_type: MarketType = MarketType.FUTURES
    taker_fee_rate: float = 0.0005

    entry_price: float | None = None
    fill_timeout_seconds: int = 3600
    entry_delay_seconds: int | None = None
    max_holding_hours: int = 72

    size_multiplier: float = 1.0

    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tp_pct is None and self.tp_price is None:
            raise ValueError("at least one of tp_pct or tp_price is required")
        if self.sl_pct is None and self.sl_price is None:
            raise ValueError("at least one of sl_pct or sl_price is required")
        if self.entry_delay_seconds is not None and self.entry_delay_seconds < 0:
            raise ValueError("entry_delay_seconds cannot be negative")
        if self.max_holding_hours <= 0:
            raise ValueError("max_holding_hours must be positive")


@dataclass(slots=True, frozen=True)
class AggTrade:
    trade_id: int
    timestamp: datetime
    price: float
    quantity: float


@dataclass(slots=True, frozen=True)
class Candle:
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    taker_buy_volume: float = 0.0


@dataclass(slots=True, frozen=True)
class ExitResolution:
    reason: ExitReason
    exit_time: datetime
    exit_price: float
    resolution_level: ResolutionLevel
    used_fallback: bool = False
    random_resolved: bool = False


@dataclass(slots=True, frozen=True)
class TradeResult:
    signal: Signal
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: ExitReason
    resolution_level: ResolutionLevel
    tp_price: float
    sl_price: float
    pnl_pct: float
    gross_pnl_pct: float
    fee_drag_pct: float
    used_fallback: bool = False
    random_resolved: bool = False


@dataclass(slots=True)
class BacktestResult:
    trades: list[TradeResult]
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    open_trades: int = 0
    unfilled: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
