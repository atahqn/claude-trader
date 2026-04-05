from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from marketdata import MarketDataRequest

if TYPE_CHECKING:
    from backtester.models import Signal
    from backtester.pipeline import PreparedMarketContext

    from .auth_client import LiveMarketClient


class FatalSignalError(RuntimeError):
    """Non-recoverable signal generator failure that should stop the engine."""


class SignalGenerator(ABC):
    """Base class for live trading signal generators.

    Implement ``poll`` with your strategy logic.  The engine calls ``setup``
    once before the main loop and ``teardown`` on shutdown.

    Subclasses **must** override the ``symbols`` property to declare which
    tickers they trade.  The engine validates disjoint symbol spaces at
    startup and filters signals against declared symbols at runtime.
    """

    analysis_interval: str = "1h"
    poll_interval: str | None = None

    @property
    def strategy_id(self) -> str:
        """Unique identifier for this generator in a multi-generator engine.

        Defaults to the class name.  Override to customise.
        """
        return type(self).__name__

    @property
    def symbols(self) -> list[str]:
        """Symbols this generator trades.  Must be overridden."""
        raise NotImplementedError(
            f"{type(self).__name__} must define a 'symbols' property"
        )

    def setup(self, client: LiveMarketClient) -> None:
        """Called once before the main loop.

        Use this to warm up indicators, fetch initial data, etc.
        Override if needed; default is a no-op.
        """

    @property
    def effective_poll_interval(self) -> str:
        return self.poll_interval or self.analysis_interval

    def set_poll_time(self, now: datetime) -> None:
        self._current_poll_time = now

    def current_time(self) -> datetime:
        """Return the engine-injected poll time.

        Strategies **must** use this instead of ``datetime.now()`` so that
        all generators in a multi-slot engine share a consistent clock.
        """
        current = getattr(self, "_current_poll_time", None)
        if current is not None:
            return current
        return datetime.now(UTC)

    def market_data_request(self) -> MarketDataRequest:
        """Declare the datasets this strategy needs.

        The default stays on the current fast path: OHLCV only at the analysis
        interval. Strategies can opt into more frequent signal checks by
        setting ``poll_interval`` to a lower aligned interval.

        Futures-native datasets are opt-in per strategy.
        """
        return MarketDataRequest.ohlcv_only(
            interval=self.analysis_interval,
            poll_interval=(
                None
                if self.effective_poll_interval == self.analysis_interval
                else self.effective_poll_interval
            ),
        )

    @property
    def cooldown_hours(self) -> float:
        """Minimum hours between signals for the same symbol.

        Used by the parallel evaluator to enforce cooldown across chunk
        boundaries.  Override if your strategy enforces a cooldown.
        """
        return 0.0

    @abstractmethod
    def poll(self) -> Signal | list[Signal] | None:
        """Return new signal(s), or ``None`` when the strategy has nothing to do."""

    def generate_backtest_signals(
        self,
        prepared_context: PreparedMarketContext,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[Signal]:
        """Generate signals for backtesting over a prepared data window.

        Override with a vectorized implementation for performance.
        Strategies used with ``StrategyEvaluator`` must implement this.

        The *prepared_context* may contain data outside ``[start, end)``
        (for indicator warmup or trade resolution).  Implementations must
        only emit signals with ``signal_date`` in ``[start, end)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_backtest_signals"
        )

    def teardown(self) -> None:
        """Called on engine shutdown.  Override for cleanup; default is a no-op."""


class CompositeSignalGenerator(SignalGenerator):
    """Wraps multiple generators for backtesting as a single unit.

    Each child generator must have a disjoint symbol space.  Signals from
    all children are merged chronologically.  Used by ``run_strategy_eval.py``
    to evaluate multi-strategy portfolios through the existing
    ``StrategyEvaluator`` without any evaluator changes.
    """

    def __init__(self, generators: list[SignalGenerator]) -> None:
        if not generators:
            raise ValueError("CompositeSignalGenerator requires at least one generator")
        self._generators = list(generators)
        self._validate_symbol_space()
        self._all_symbols = []
        for gen in self._generators:
            self._all_symbols.extend(gen.symbols)

    def _validate_symbol_space(self) -> None:
        seen: dict[str, str] = {}
        for gen in self._generators:
            for symbol in gen.symbols:
                if symbol in seen:
                    raise ValueError(
                        f"Symbol space conflict: '{symbol}' claimed by both "
                        f"'{seen[symbol]}' and '{gen.strategy_id}'"
                    )
                seen[symbol] = gen.strategy_id

    @property
    def strategy_id(self) -> str:
        return "+".join(g.strategy_id for g in self._generators)

    @property
    def symbols(self) -> list[str]:
        return list(self._all_symbols)

    def poll(self) -> Signal | list[Signal] | None:
        return None

    def generate_backtest_signals(
        self,
        prepared_context: PreparedMarketContext,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[Signal]:
        all_signals: list[Signal] = []
        for gen in self._generators:
            child_symbols = [s for s in symbols if s in gen.symbols]
            if not child_symbols:
                continue
            sigs = gen.generate_backtest_signals(
                prepared_context, child_symbols, start, end,
            )
            all_signals.extend(sigs)
        all_signals.sort(key=lambda s: s.signal_date)
        return all_signals
