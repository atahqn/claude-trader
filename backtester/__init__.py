"""Signal backtesting framework for crypto trading strategies."""

from .models import (
    AggTrade,
    BacktestResult,
    Candle,
    ExitReason,
    ExitResolution,
    MarketType,
    PositionType,
    ResolutionLevel,
    Signal,
    TradeResult,
)
from .data import BinanceClient
from .engine import backtest_signal, backtest_signals
from .pipeline import (
    BacktestExecutionSession,
    PreparedMarketContext,
    prepare_market_context,
)
from .eval_windows import (
    ALL_WINDOWS,
    DEVELOPMENT_WINDOWS,
    EVALUATION_WINDOWS,
    EvalWindow,
    HOLDOUT_WINDOWS,
    LEGACY_DEVELOPMENT_WINDOWS,
    OOS2_WINDOWS,
    STRESS_DEVELOPMENT_WINDOWS,
)
from .evaluator import (
    CategorySummary,
    EvaluationReport,
    PortfolioConfig,
    StrategyEvaluator,
    WindowResult,
)
from .resolver import compute_pnl, compute_tp_sl_prices
from .validation import LookaheadViolation, validate_no_lookahead

__all__ = [
    "AggTrade",
    "BacktestResult",
    "BacktestExecutionSession",
    "BinanceClient",
    "Candle",
    "ExitReason",
    "ExitResolution",
    "MarketType",
    "PositionType",
    "PreparedMarketContext",
    "ResolutionLevel",
    "Signal",
    "TradeResult",
    "ALL_WINDOWS",
    "CategorySummary",
    "DEVELOPMENT_WINDOWS",
    "EVALUATION_WINDOWS",
    "EvalWindow",
    "EvaluationReport",
    "HOLDOUT_WINDOWS",
    "LEGACY_DEVELOPMENT_WINDOWS",
    "LookaheadViolation",
    "OOS2_WINDOWS",
    "PortfolioConfig",
    "STRESS_DEVELOPMENT_WINDOWS",
    "StrategyEvaluator",
    "WindowResult",
    "backtest_signal",
    "backtest_signals",
    "compute_pnl",
    "compute_tp_sl_prices",
    "prepare_market_context",
    "validate_no_lookahead",
]
