"""Signal backtesting framework for crypto trading strategies."""

from .models import (
    AggTrade,
    BacktestResult,
    Candle,
    ExitReason,
    ExitResolution,
    MarketType,
    PortfolioResult,
    PositionType,
    ResolutionLevel,
    Signal,
    SkipReason,
    SkippedSignal,
    TradeResult,
)
from .data import BinanceClient
from .engine import backtest_portfolio, backtest_signal, backtest_signals
from .resolver import compute_pnl, compute_tp_sl_prices

__all__ = [
    "AggTrade",
    "BacktestResult",
    "BinanceClient",
    "Candle",
    "ExitReason",
    "ExitResolution",
    "MarketType",
    "PortfolioResult",
    "PositionType",
    "ResolutionLevel",
    "Signal",
    "SkipReason",
    "SkippedSignal",
    "TradeResult",
    "backtest_portfolio",
    "backtest_signal",
    "backtest_signals",
    "compute_pnl",
    "compute_tp_sl_prices",
]
