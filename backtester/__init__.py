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
from .pipeline import (
    BacktestExecutionSession,
    PreparedMarketContext,
    generate_signals_from_prepared_context,
    prepare_market_context,
)
from .resolver import compute_pnl, compute_tp_sl_prices

__all__ = [
    "AggTrade",
    "BacktestResult",
    "BacktestExecutionSession",
    "BinanceClient",
    "Candle",
    "ExitReason",
    "ExitResolution",
    "MarketType",
    "PortfolioResult",
    "PositionType",
    "PreparedMarketContext",
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
    "generate_signals_from_prepared_context",
    "prepare_market_context",
]
