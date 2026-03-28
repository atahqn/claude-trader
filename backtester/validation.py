"""Look-ahead bias validation for signal generators.

The validator truncates market data to a signal's timestamp and re-runs
signal generation.  If the signal disappears when future data is removed,
it was produced using information that did not exist yet.

This is a one-time check per strategy version, not a per-backtest guard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .models import Signal

if TYPE_CHECKING:
    from live.signal_generator import SignalGenerator

    from .pipeline import PreparedMarketContext


@dataclass(frozen=True, slots=True)
class LookaheadViolation:
    signal: Signal
    detail: str


def validate_no_lookahead(
    generator: SignalGenerator,
    prepared_context: PreparedMarketContext,
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> list[LookaheadViolation]:
    """Verify that no signal depends on future data.

    Generates all signals on the full window, then for a random sample
    truncates the context data to ``signal_date`` and re-generates with
    the **full** symbol universe and the **original** ``start``/``end``.

    If a signal disappears or changes parameters when future data is
    removed, it had look-ahead bias.

    Scope: validates that the same ``Signal`` object (including TP/SL)
    is produced.  Does not cover post-generation adjustments (trailing
    stops, dynamic leverage changes) which are outside signal generation.

    Parameters
    ----------
    generator:
        Must implement ``generate_backtest_signals``.
    prepared_context:
        Full market data for the evaluation window.
    symbols:
        Symbol universe.  Re-used in full for every replay to avoid
        false positives from cross-symbol logic.
    start, end:
        Signal generation window (exclusive end).
    sample_size:
        Number of signals to validate.  More is slower but more thorough.
    seed:
        RNG seed for reproducible sampling.
    """
    all_signals = generator.generate_backtest_signals(
        prepared_context, symbols, start, end,
    )
    if not all_signals:
        return []

    rng = random.Random(seed)
    sampled = rng.sample(all_signals, min(len(all_signals), sample_size))

    violations: list[LookaheadViolation] = []
    for signal in sampled:
        truncated_ctx = prepared_context.truncated_to(signal.signal_date)
        trunc_signals = generator.generate_backtest_signals(
            truncated_ctx, symbols, start, end,
        )
        if not _signal_present(signal, trunc_signals):
            violations.append(
                LookaheadViolation(
                    signal=signal,
                    detail=(
                        f"Signal at {signal.signal_date} for {signal.ticker} "
                        f"{signal.position_type.name} disappeared when data "
                        f"after {signal.signal_date} was removed"
                    ),
                )
            )
    return violations


def _signal_present(target: Signal, candidates: list[Signal]) -> bool:
    """Check if *target* exists in *candidates* on key fields."""
    for s in candidates:
        if (
            s.signal_date == target.signal_date
            and s.ticker == target.ticker
            and s.position_type == target.position_type
            and s.tp_pct == target.tp_pct
            and s.sl_pct == target.sl_pct
        ):
            return True
    return False
