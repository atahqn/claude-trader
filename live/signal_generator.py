from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from marketdata import MarketDataRequest

if TYPE_CHECKING:
    from backtester.models import Signal

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

    def teardown(self) -> None:
        """Called on engine shutdown.  Override for cleanup; default is a no-op."""
