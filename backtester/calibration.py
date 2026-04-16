"""Adaptive parameter calibration for signal generators."""

from __future__ import annotations

import itertools
import logging
import multiprocessing
import os
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


# Minimum grid size to justify process-pool overhead.
_PARALLEL_MIN_COMBOS = 4

# Module-level refs set before forking so children inherit them.
_shared_score_fn: Callable | None = None
_shared_frame: pd.DataFrame | None = None


def _init_worker(score_fn: Callable, frame: pd.DataFrame) -> None:
    """Initializer for pool workers — stash refs inherited via fork."""
    global _shared_score_fn, _shared_frame
    _shared_score_fn = score_fn
    _shared_frame = frame


def _score_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
    """Worker: score one parameter combo against inherited score_fn/frame."""
    try:
        return candidate, _shared_score_fn(candidate, _shared_frame)
    except Exception:
        return candidate, None


def search_parameters(
    param_space: dict[str, list],
    score_fn: Callable[[dict, pd.DataFrame], float],
    frame: pd.DataFrame,
    *,
    max_workers: int = 0,
    prepare_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> CalibrationResult | None:
    """Grid search over *param_space*, return the best-scoring combination.

    When *prepare_fn* is provided it is called once on a shallow copy of
    *frame* before any scoring begins, allowing shared state to be
    pre-computed and attached to ``frame.attrs``.

    When *max_workers* is 0 (the default) and the grid has at least
    ``_PARALLEL_MIN_COMBOS`` entries, scoring runs in parallel across
    processes to bypass the GIL.  Set *max_workers* to 1 to force
    sequential execution.

    Uses ``fork``-based multiprocessing so *score_fn* does not need to be
    picklable — it is inherited by child processes.

    Returns ``None`` if *param_space* is empty, *frame* is empty, or every
    candidate raises an exception.
    """
    if not param_space or frame.empty:
        return None

    if prepare_fn is not None:
        work = frame.copy(deep=False)
        prepared = prepare_fn(work)
        if not isinstance(prepared, pd.DataFrame):
            raise TypeError(
                f"prepare_fn must return a DataFrame, got {type(prepared).__name__}"
            )
        frame = prepared
        if frame.empty:
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

    candidates = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Decide parallelism — only use fork (the Linux default).
    can_fork = multiprocessing.get_start_method() == "fork"
    if max_workers == 1 or total < _PARALLEL_MIN_COMBOS or not can_fork:
        effective_workers = 1
    elif max_workers == 0:
        effective_workers = min(total, os.cpu_count() or 4)
    else:
        effective_workers = min(total, max_workers)

    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    evaluated = 0

    if effective_workers <= 1:
        for candidate in candidates:
            try:
                score = score_fn(candidate, frame)
            except Exception:
                logger.warning("score_params raised for %s — skipping", candidate)
                continue
            evaluated += 1
            if score > best_score:
                best_score = score
                best_params = candidate
    else:
        # Set module-level refs so fork'd children inherit them.
        global _shared_score_fn, _shared_frame
        _shared_score_fn = score_fn
        _shared_frame = frame
        try:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=effective_workers) as pool:
                results = pool.map(_score_candidate, candidates)
            for candidate, score in results:
                if score is None:
                    logger.warning(
                        "score_params raised for %s — skipping", candidate
                    )
                    continue
                evaluated += 1
                if score > best_score:
                    best_score = score
                    best_params = candidate
        except Exception:
            logger.debug("parallel calibration failed — falling back to sequential")
            for candidate in candidates:
                try:
                    score = score_fn(candidate, frame)
                except Exception:
                    logger.warning(
                        "score_params raised for %s — skipping", candidate
                    )
                    continue
                evaluated += 1
                if score > best_score:
                    best_score = score
                    best_params = candidate
        finally:
            _shared_score_fn = None
            _shared_frame = None

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
