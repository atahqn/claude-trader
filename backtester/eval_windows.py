"""Evaluation window definitions for the widened weekly calendar.

All dates match STRATEGY_EVOLUTION.md.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True, slots=True)
class EvalWindow:
    name: str
    start: datetime
    end: datetime
    category: str


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=UTC)


def _weekly_block(
    prefix: str,
    category: str,
    start: datetime,
    count: int,
) -> list[EvalWindow]:
    windows: list[EvalWindow] = []
    cursor = start
    for idx in range(1, count + 1):
        end = cursor + timedelta(days=7)
        windows.append(EvalWindow(f"{prefix}_W{idx}", cursor, end, category))
        cursor = end
    return windows


# ---------------------------------------------------------------------------
# Legacy development windows (24)
# ---------------------------------------------------------------------------

LEGACY_DEVELOPMENT_WINDOWS: list[EvalWindow] = [
    *_weekly_block("Apr24", "development", _dt(2024, 4, 15), 4),
    *_weekly_block("Aug24", "development", _dt(2024, 8, 1), 4),
    *_weekly_block("Oct24", "development", _dt(2024, 10, 15), 4),
    *_weekly_block("Dec24", "development", _dt(2024, 12, 1), 4),
    *_weekly_block("JanFeb25", "development", _dt(2025, 1, 15), 4),
    *_weekly_block("Mar25", "development", _dt(2025, 3, 1), 4),
]

# ---------------------------------------------------------------------------
# Added development stress weeks (4)
# ---------------------------------------------------------------------------

STRESS_DEVELOPMENT_WINDOWS: list[EvalWindow] = [
    EvalWindow("DEVX_MAY24_BULL", _dt(2024, 5, 14), _dt(2024, 5, 21), "development_stress"),
    EvalWindow("DEVX_NOV24_BULL", _dt(2024, 11, 15), _dt(2024, 11, 22), "development_stress"),
    EvalWindow("DEVX_FEB25_BEAR", _dt(2025, 2, 20), _dt(2025, 2, 27), "development_stress"),
    EvalWindow("DEVX_AUG25_CHOP", _dt(2025, 8, 20), _dt(2025, 8, 27), "development_stress"),
]

# ---------------------------------------------------------------------------
# Combined development (28)
# ---------------------------------------------------------------------------

DEVELOPMENT_WINDOWS: list[EvalWindow] = (
    LEGACY_DEVELOPMENT_WINDOWS + STRESS_DEVELOPMENT_WINDOWS
)

# ---------------------------------------------------------------------------
# Primary holdout windows (7)
# ---------------------------------------------------------------------------

HOLDOUT_WINDOWS: list[EvalWindow] = [
    *_weekly_block("Apr25", "holdout", _dt(2025, 4, 1), 4),
    *_weekly_block("OOS26", "holdout", _dt(2026, 3, 1), 3),
]

# ---------------------------------------------------------------------------
# Secondary OOS hard-week pack (4)
# ---------------------------------------------------------------------------

OOS2_WINDOWS: list[EvalWindow] = [
    EvalWindow("OOS2_BULL25", _dt(2025, 5, 6), _dt(2025, 5, 13), "oos2"),
    EvalWindow("OOS2_BEAR25", _dt(2025, 10, 3), _dt(2025, 10, 10), "oos2"),
    EvalWindow("OOS2_CHOP25", _dt(2025, 11, 27), _dt(2025, 12, 4), "oos2"),
    EvalWindow("OOS2_CAPIT26", _dt(2026, 1, 29), _dt(2026, 2, 5), "oos2"),
]

# ---------------------------------------------------------------------------
# Combined evaluation (11) and full calendar (39)
# ---------------------------------------------------------------------------

EVALUATION_WINDOWS: list[EvalWindow] = HOLDOUT_WINDOWS + OOS2_WINDOWS

ALL_WINDOWS: list[EvalWindow] = DEVELOPMENT_WINDOWS + EVALUATION_WINDOWS


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def windows_by_category(
    windows: list[EvalWindow],
) -> dict[str, list[EvalWindow]]:
    groups: dict[str, list[EvalWindow]] = defaultdict(list)
    for w in windows:
        groups[w.category].append(w)
    return dict(groups)


def date_range(windows: list[EvalWindow]) -> tuple[datetime, datetime]:
    """Return (earliest start, latest end) across *windows*."""
    return min(w.start for w in windows), max(w.end for w in windows)
