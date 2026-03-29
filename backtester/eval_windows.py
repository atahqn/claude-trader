"""Evaluation window definitions for the widened weekly calendar.

This is the canonical source of truth for all active evaluation windows.
STRATEGY_EVOLUTION.md references this file; do not duplicate date tables there.

Window selection method
-----------------------
Windows are selected by **market regime characteristics**, never by strategy
performance. The original liquid 8-symbol basket (BTC, ETH, SOL, BNB, XRP,
DOGE, AVAX, LINK) is used for regime classification so the full Jul-2023 to
Mar-2026 history is covered consistently.

For each candidate 7-day window, these metrics are computed:
  - Average basket return (bull/bear/flat)
  - Up-breadth (% of symbols with positive return)
  - Cross-sectional dispersion (how differently symbols move)
  - Realized hourly volatility (annualized)
  - Intra-week max drawdown

Windows are chosen to maximise regime diversity: broad bull impulse, broad
bear selloff, high-volatility chop, high-dispersion divergence, and
capitulation events. See ``alpha_lab/scan_new_windows.py`` for the scanning
tool used to identify candidates.

Calendar summary
----------------
  38 development windows  (24 legacy + 10 stress + 4 bullish)
  21 evaluation windows   (7 primary holdout + 4 secondary OOS + 6 tertiary OOS + 4 bull OOS)
  59 total windows        spanning Nov 2020 – Mar 2026
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
# Six 4-week blocks sampling different market regimes across Apr 2024 – Mar 2025.
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
# Development stress weeks (10)
# Hand-picked for hard market structure: crashes, high-vol chop,
# high-dispersion divergence, and bear selloffs. Includes pre-2024 and
# gap-period weeks to force development beyond the original date range.
#
#   DEVX_MAY24_BULL    — broad bull impulse
#   DEVX_NOV24_BULL    — high-beta bull breakout
#   DEVX_FEB25_BEAR    — broad bear selloff
#   DEVX_AUG25_CHOP    — low-drift, high-noise chop
#   DEVX_AUG23_CRASH   — pre-2024 crash (−14.27% avg return, 0% breadth)
#   DEVX_DEC23_DISP    — pre-2024 high-dispersion chop (11.62 disp, 92.3 vol)
#   DEVX_JAN24_CHOP    — pre-2024 high-vol chop (95.6 vol, 10.49% DD)
#   DEVX_MAR24_HV      — pre-2024 bear, highest vol (122.4), 17.26% DD
#   DEVX_JUN24_CRASH   — gap-period crash (−10.43%, 21.36% DD)
#   DEVX_SEP24_BEAR    — gap-period bear selloff (−10.42%, 17.19% DD)
# ---------------------------------------------------------------------------

STRESS_DEVELOPMENT_WINDOWS: list[EvalWindow] = [
    EvalWindow("DEVX_MAY24_BULL", _dt(2024, 5, 14), _dt(2024, 5, 21), "development_stress"),
    EvalWindow("DEVX_NOV24_BULL", _dt(2024, 11, 15), _dt(2024, 11, 22), "development_stress"),
    EvalWindow("DEVX_FEB25_BEAR", _dt(2025, 2, 20), _dt(2025, 2, 27), "development_stress"),
    EvalWindow("DEVX_AUG25_CHOP", _dt(2025, 8, 20), _dt(2025, 8, 27), "development_stress"),
    EvalWindow("DEVX_AUG23_CRASH", _dt(2023, 8, 12), _dt(2023, 8, 19), "development_stress"),
    EvalWindow("DEVX_DEC23_DISP", _dt(2023, 12, 9), _dt(2023, 12, 16), "development_stress"),
    EvalWindow("DEVX_JAN24_CHOP", _dt(2024, 1, 6), _dt(2024, 1, 13), "development_stress"),
    EvalWindow("DEVX_MAR24_HV", _dt(2024, 3, 16), _dt(2024, 3, 23), "development_stress"),
    EvalWindow("DEVX_JUN24_CRASH", _dt(2024, 6, 29), _dt(2024, 7, 6), "development_stress"),
    EvalWindow("DEVX_SEP24_BEAR", _dt(2024, 9, 28), _dt(2024, 10, 5), "development_stress"),
]

# ---------------------------------------------------------------------------
# Development bullish weeks (4)
# Selected from Oct-2020 to Mar-2023 using the regime basket only, with
# emphasis on broad bullish weeks that add bull-market coverage to the
# otherwise crash/chop-heavy development set.
#   DEVB_NOV20_BREAKOUT  — early cycle breakout (+31.16%, 100% breadth)
#   DEVB_JAN21_MANIA     — peak broad mania (+80.37%, 100% breadth)
#   DEVB_AUG21_REBOUND   — post-summer squeeze rebound (+32.02%, 100% breadth)
#   DEVB_JUL22_RALLY     — bear-market countertrend rally (+28.62%, 100%)
# ---------------------------------------------------------------------------

BULL_DEVELOPMENT_WINDOWS: list[EvalWindow] = [
    EvalWindow("DEVB_NOV20_BREAKOUT", _dt(2020, 11, 18), _dt(2020, 11, 25), "development_bull"),
    EvalWindow("DEVB_JAN21_MANIA", _dt(2021, 1, 28), _dt(2021, 2, 4), "development_bull"),
    EvalWindow("DEVB_AUG21_REBOUND", _dt(2021, 8, 9), _dt(2021, 8, 16), "development_bull"),
    EvalWindow("DEVB_JUL22_RALLY", _dt(2022, 7, 13), _dt(2022, 7, 20), "development_bull"),
]

# ---------------------------------------------------------------------------
# Combined development (38)
# ---------------------------------------------------------------------------

DEVELOPMENT_WINDOWS: list[EvalWindow] = (
    LEGACY_DEVELOPMENT_WINDOWS + STRESS_DEVELOPMENT_WINDOWS + BULL_DEVELOPMENT_WINDOWS
)

# ---------------------------------------------------------------------------
# Primary holdout windows (7)
# Apr25: 4 contiguous weeks in a neutral-to-mild-bull period.
# OOS26: 3 contiguous weeks near the end of the data range.
# ---------------------------------------------------------------------------

HOLDOUT_WINDOWS: list[EvalWindow] = [
    *_weekly_block("Apr25", "holdout", _dt(2025, 4, 1), 4),
    *_weekly_block("OOS26", "holdout", _dt(2026, 3, 1), 3),
]

# ---------------------------------------------------------------------------
# Secondary OOS hard-week pack (4)
# Intentionally composed of hard market structure weeks:
#   OOS2_BULL25   — broad bull impulse
#   OOS2_BEAR25   — fast broad liquidation
#   OOS2_CHOP25   — low-drift choppy week
#   OOS2_CAPIT26  — capitulation bear week
# ---------------------------------------------------------------------------

OOS2_WINDOWS: list[EvalWindow] = [
    EvalWindow("OOS2_BULL25", _dt(2025, 5, 6), _dt(2025, 5, 13), "oos2"),
    EvalWindow("OOS2_BEAR25", _dt(2025, 10, 3), _dt(2025, 10, 10), "oos2"),
    EvalWindow("OOS2_CHOP25", _dt(2025, 11, 27), _dt(2025, 12, 4), "oos2"),
    EvalWindow("OOS2_CAPIT26", _dt(2026, 1, 29), _dt(2026, 2, 5), "oos2"),
]

# ---------------------------------------------------------------------------
# Tertiary OOS: pre-2024 and gap-period hard weeks (6)
# Selected by market regime characteristics (not strategy performance):
#   OOS3_JUL23_DISP   — bull + extreme cross-symbol dispersion (16.35), pre-2024
#   OOS3_NOV23_DISP   — mild bull + highest dispersion (18.58), pre-2024
#   OOS3_JUL24_CRASH  — −11.59% bear selloff, gap period
#   OOS3_AUG24_BEAR   — −10.00% uniform bear, 28.6% baseline trade WR
#   OOS3_OCT25_HV     — mild bear + highest realized vol (103.9), gap period
#   OOS3_NOV25_CRASH  — −11.81% bear selloff, gap period
# ---------------------------------------------------------------------------

OOS3_WINDOWS: list[EvalWindow] = [
    EvalWindow("OOS3_JUL23_DISP", _dt(2023, 7, 8), _dt(2023, 7, 15), "oos3"),
    EvalWindow("OOS3_NOV23_DISP", _dt(2023, 11, 11), _dt(2023, 11, 18), "oos3"),
    EvalWindow("OOS3_JUL24_CRASH", _dt(2024, 7, 20), _dt(2024, 7, 27), "oos3"),
    EvalWindow("OOS3_AUG24_BEAR", _dt(2024, 8, 31), _dt(2024, 9, 7), "oos3"),
    EvalWindow("OOS3_OCT25_HV", _dt(2025, 10, 11), _dt(2025, 10, 18), "oos3"),
    EvalWindow("OOS3_NOV25_CRASH", _dt(2025, 11, 15), _dt(2025, 11, 22), "oos3"),
]

# ---------------------------------------------------------------------------
# Quaternary OOS: pre-Jul-2023 bull impulse pack (4)
# Selected from Oct-2020 to Mar-2023 using the regime basket only, with
# emphasis on broad bullish weeks (100% up-breadth) that widen coverage beyond
# the current crash/chop-heavy holdout packs.
#   OOS4_JAN21_BREAKOUT   — broad BTC/ETH-led breakout (+50.86%, 100% breadth)
#   OOS4_FEB21_MANIA      — broad high-beta mania (+76.45%, 100% breadth)
#   OOS4_APR21_ALT        — broad alt rotation impulse (+46.43%, 100% breadth)
#   OOS4_JAN23_REPRICING  — post-bear-market repricing rally (+31.80%, 100%)
# ---------------------------------------------------------------------------

OOS4_WINDOWS: list[EvalWindow] = [
    EvalWindow("OOS4_JAN21_BREAKOUT", _dt(2021, 1, 1), _dt(2021, 1, 8), "oos4"),
    EvalWindow("OOS4_FEB21_MANIA", _dt(2021, 2, 4), _dt(2021, 2, 11), "oos4"),
    EvalWindow("OOS4_APR21_ALT", _dt(2021, 4, 9), _dt(2021, 4, 16), "oos4"),
    EvalWindow("OOS4_JAN23_REPRICING", _dt(2023, 1, 8), _dt(2023, 1, 15), "oos4"),
]

# ---------------------------------------------------------------------------
# Combined evaluation (21) and full calendar (59)
# ---------------------------------------------------------------------------

EVALUATION_WINDOWS: list[EvalWindow] = HOLDOUT_WINDOWS + OOS2_WINDOWS + OOS3_WINDOWS + OOS4_WINDOWS

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
