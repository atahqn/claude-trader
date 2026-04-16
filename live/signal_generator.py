from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

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

    # -- Adaptive calibration (opt-in) ----------------------------------------
    needs_calibration: bool = False
    calibration_interval_hours: int = 168      # how often to recalibrate
    calibration_lookback_hours: int = 720      # max history for calibration

    @property
    def active_params(self) -> dict[str, Any]:
        """Current calibrated parameters, set by the framework."""
        return getattr(self, "_active_params", {})

    @active_params.setter
    def active_params(self, value: dict[str, Any]) -> None:
        self._active_params = value

    def param_space(self) -> dict[str, list]:
        """Search space for calibration.

        Return a dict mapping parameter names to candidate value lists.
        Example: ``{"tp_pct": [2.0, 2.5, 3.0], "rsi_thresh": [25, 30, 35]}``

        Only called when ``needs_calibration`` is ``True``.
        """
        return {}

    def score_params(self, params: dict[str, Any], frame: pd.DataFrame) -> float:
        """Score a candidate parameter set on historical data.

        Called by the calibration search for each combination from
        :meth:`param_space`.  *frame* is a DataFrame covering the lookback
        window (may include indicator columns if :meth:`build_calibration_frame`
        adds them).

        Return a float where **higher is better**.
        """
        return 0.0

    def prepare_score_context(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute shared state before the parameter grid search.

        Called once per calibration cycle, before ``score_params`` runs for
        each parameter combination.  The hook receives a shallow copy of the
        calibration frame and **must only write to** ``frame.attrs`` — never
        mutate columns, index, or underlying data (the shallow copy shares
        column buffers with the original).  Frame enrichment (adding columns,
        computing indicators) belongs in ``build_calibration_frame``, not here.

        Cached attrs and any objects they reference **must not be mutated**
        by ``score_params`` — they are shared across all candidate
        evaluations and across forked processes in parallel mode, where
        mutation defeats copy-on-write.

        Must return the (enriched) DataFrame.  Default: returns *frame*
        unchanged.
        """
        return frame

    def build_calibration_frame(
        self, frame: pd.DataFrame, t: datetime,
    ) -> pd.DataFrame:
        """Filter / enrich the lookback data before calibration scoring.

        *frame* contains the last ``calibration_lookback_hours`` of OHLCV data
        across all symbols (with a ``symbol`` column), where every row has
        ``close_time < t``.

        Override to select a regime subset, add indicators, etc.
        Default: returns *frame* unchanged.
        """
        return frame

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

    @property
    def required_warmup_bars(self) -> int:
        """Minimum number of warmup bars the strategy needs.

        The evaluator fetches this many extra bars before the signal window
        so that indicators have enough history.  If ``indicator_request()``
        is also declared, the framework ensures at least
        ``required_warmup(indicator_request())`` bars automatically; set
        this higher only when you need additional bars beyond the indicator
        minimum.  Default is 100 for backward compatibility.
        """
        return 100

    def indicator_request(self) -> tuple[str, ...]:
        """Declare indicator columns to precompute on each symbol's OHLCV frame.

        Return indicator names recognised by ``compute_indicator_frame()``.
        The framework precomputes these in ``PreparedMarketContext`` and
        makes them available via ``ctx.indicator_frame(symbol)``.

        When declared, the framework automatically ensures enough warmup
        bars are fetched for the requested indicators.
        """
        return ()

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

    @property
    def needs_calibration(self) -> bool:  # type: ignore[override]
        return any(g.needs_calibration for g in self._generators)

    @property
    def calibration_interval_hours(self) -> int:  # type: ignore[override]
        children = self.calibration_children()
        return children[0].calibration_interval_hours if children else 168

    @property
    def calibration_lookback_hours(self) -> int:  # type: ignore[override]
        children = self.calibration_children()
        return children[0].calibration_lookback_hours if children else 720

    def calibration_children(self) -> list[SignalGenerator]:
        """Return child generators that require calibration."""
        return [g for g in self._generators if g.needs_calibration]

    def __init__(self, generators: list[SignalGenerator]) -> None:
        if not generators:
            raise ValueError("CompositeSignalGenerator requires at least one generator")
        self._generators = list(generators)
        self._validate_symbol_space()
        self._validate_calibration_agreement()
        self._all_symbols: list[str] = []
        for gen in self._generators:
            self._all_symbols.extend(gen.symbols)

    def _validate_calibration_agreement(self) -> None:
        """All calibrating children must share the same interval and lookback."""
        children = self.calibration_children()
        if len(children) <= 1:
            return
        first = children[0]
        for child in children[1:]:
            if child.calibration_interval_hours != first.calibration_interval_hours:
                raise ValueError(
                    f"Calibration interval conflict in CompositeSignalGenerator: "
                    f"'{first.strategy_id}' uses {first.calibration_interval_hours}h "
                    f"but '{child.strategy_id}' uses {child.calibration_interval_hours}h. "
                    f"All calibrating children must share the same calibration_interval_hours."
                )
            if child.calibration_lookback_hours != first.calibration_lookback_hours:
                raise ValueError(
                    f"Calibration lookback conflict in CompositeSignalGenerator: "
                    f"'{first.strategy_id}' uses {first.calibration_lookback_hours}h "
                    f"but '{child.strategy_id}' uses {child.calibration_lookback_hours}h. "
                    f"All calibrating children must share the same calibration_lookback_hours."
                )

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
    def required_warmup_bars(self) -> int:
        return max((g.required_warmup_bars for g in self._generators), default=100)

    def indicator_request(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for g in self._generators:
            for ind in g.indicator_request():
                seen.setdefault(ind, None)
        return tuple(seen)

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
