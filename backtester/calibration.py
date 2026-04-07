"""Adaptive parameter calibration for signal generators."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    from live.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

_MAX_COMBINATIONS = 50_000


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    best_params: dict[str, Any]
    best_score: float
    candidates_evaluated: int


def search_parameters(
    param_space: dict[str, list],
    score_fn: Callable[[dict, pd.DataFrame], float],
    frame: pd.DataFrame,
) -> CalibrationResult | None:
    """Grid search over *param_space*, return the best-scoring combination.

    Returns ``None`` if *param_space* is empty, *frame* is empty, or every
    candidate raises an exception.
    """
    if not param_space or frame.empty:
        return None

    keys = list(param_space.keys())
    values = list(param_space.values())

    total = 1
    for v in values:
        total *= len(v)
    if total > _MAX_COMBINATIONS:
        raise ValueError(
            f"param_space produces {total:,} combinations "
            f"(limit is {_MAX_COMBINATIONS:,}). Reduce the search space."
        )

    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    evaluated = 0

    for combo in itertools.product(*values):
        candidate = dict(zip(keys, combo))
        try:
            score = score_fn(candidate, frame)
        except Exception:
            logger.warning("score_params raised for %s — skipping", candidate)
            continue
        evaluated += 1
        if score > best_score:
            best_score = score
            best_params = candidate

    if evaluated == 0:
        return None

    return CalibrationResult(
        best_params=best_params,
        best_score=best_score,
        candidates_evaluated=evaluated,
    )


def validate_calibration_config(generator: SignalGenerator) -> None:
    """Raise ``ValueError`` if a calibrating generator is misconfigured."""
    if not generator.needs_calibration:
        return

    from live.signal_generator import CompositeSignalGenerator

    if isinstance(generator, CompositeSignalGenerator):
        for child in generator.calibration_children():
            validate_calibration_config(child)
        return

    space = generator.param_space()
    if not space:
        raise ValueError(
            f"{type(generator).__name__} has needs_calibration=True "
            f"but param_space() returned an empty dict"
        )
    for key, vals in space.items():
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(
                f"{type(generator).__name__}.param_space()['{key}'] "
                f"must be a non-empty list"
            )
