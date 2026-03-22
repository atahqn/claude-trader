from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtester.models import Signal

    from .auth_client import LiveMarketClient


class SignalGenerator(ABC):
    """Base class for live trading signal generators.

    Implement ``poll`` with your strategy logic.  The engine calls ``setup``
    once before the main loop and ``teardown`` on shutdown.
    """

    def setup(self, client: LiveMarketClient) -> None:
        """Called once before the main loop.

        Use this to warm up indicators, fetch initial data, etc.
        Override if needed; default is a no-op.
        """

    @abstractmethod
    def poll(self) -> Signal | list[Signal] | None:
        """Return new signal(s), or ``None`` when the strategy has nothing to do."""

    def teardown(self) -> None:
        """Called on engine shutdown.  Override for cleanup; default is a no-op."""
