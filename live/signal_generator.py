from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from marketdata import MarketDataRequest

if TYPE_CHECKING:
    from backtester.models import Signal
    from backtester.pipeline import PreparedMarketContext

    from .auth_client import LiveMarketClient


class SignalGenerator(ABC):
    """Base class for live trading signal generators.

    Implement ``poll`` with your strategy logic.  The engine calls ``setup``
    once before the main loop and ``teardown`` on shutdown.
    """

    analysis_interval: str = "1h"
    poll_interval: str | None = None

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
