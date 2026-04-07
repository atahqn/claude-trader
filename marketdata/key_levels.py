from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime

from backtester.models import Candle

# ---------------------------------------------------------------------------
# Session boundaries (UTC hours)
# ---------------------------------------------------------------------------
ASIA_START_HOUR = 0
ASIA_END_HOUR = 8
LONDON_START_HOUR = 8
LONDON_END_HOUR = 16
NY_START_HOUR = 13
NY_END_HOUR = 22


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class KeyLevels:
    """Structural price levels at a point in time.

    All values are ``None`` when there is insufficient history to compute them.

    Naming conventions
    ------------------
    * ``prev_*``  – from the most recently **completed** period of that timeframe.
    * ``*_open``  – opening price of the **current** period.
    * ``*_eq``    – equilibrium / midpoint  ``(high + low) / 2``.
    * ``pdh/pdl`` – Previous Day High / Previous Day Low.
    * ``asia_*``  – most recently completed Asia session (a.k.a. ONH / ONL).
    """

    # 4-Hour
    h4_open: float | None = None
    prev_h4_high: float | None = None
    prev_h4_low: float | None = None
    h4_eq: float | None = None

    # Daily
    daily_open: float | None = None
    pdh: float | None = None
    pdl: float | None = None
    daily_eq: float | None = None

    # Weekly
    weekly_open: float | None = None
    prev_week_high: float | None = None
    prev_week_low: float | None = None
    weekly_eq: float | None = None

    # Monthly
    monthly_open: float | None = None
    prev_month_high: float | None = None
    prev_month_low: float | None = None
    monthly_eq: float | None = None

    # Quarterly
    quarterly_open: float | None = None
    prev_quarter_high: float | None = None
    prev_quarter_low: float | None = None
    quarterly_eq: float | None = None

    # Yearly (running high/low of the current year)
    yearly_open: float | None = None
    yearly_high: float | None = None
    yearly_low: float | None = None
    yearly_eq: float | None = None

    # Monday range (completed Monday only — available from Tuesday)
    monday_high: float | None = None
    monday_low: float | None = None
    monday_mid: float | None = None

    # Sessions — most recently completed
    asia_open: float | None = None
    asia_high: float | None = None
    asia_low: float | None = None

    london_open: float | None = None
    london_high: float | None = None
    london_low: float | None = None

    ny_open: float | None = None
    ny_high: float | None = None
    ny_low: float | None = None


# ---------------------------------------------------------------------------
# Internal dataclasses for precomputation
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class _AggPeriod:
    start: datetime
    open: float
    high: float
    low: float


@dataclass(slots=True)
class _YearlyRunning:
    year_open: float
    candle_close_times: list[datetime]
    cum_highs: list[float]
    cum_lows: list[float]


@dataclass(slots=True, frozen=True)
class _CompletedSession:
    end_time: datetime
    open: float
    high: float
    low: float


@dataclass(slots=True, frozen=True)
class _CompletedMonday:
    end_time: datetime
    high: float
    low: float
    mid: float


# ---------------------------------------------------------------------------
# Helpers — standard timeframes (4H / D / W / M)
# ---------------------------------------------------------------------------

def _open_prev_levels(
    candles: list[Candle],
    t: datetime,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return ``(current_open, prev_high, prev_low, eq)`` via binary search."""
    if not candles:
        return None, None, None, None
    idx = bisect_right(candles, t, key=lambda c: c.open_time) - 1
    if idx < 0:
        return None, None, None, None
    current = candles[idx]
    if idx == 0:
        return current.open, None, None, None
    prev = candles[idx - 1]
    eq = (prev.high + prev.low) / 2
    return current.open, prev.high, prev.low, eq


# ---------------------------------------------------------------------------
# Helpers — quarterly (aggregated from daily candles)
# ---------------------------------------------------------------------------

def _quarter_of(dt: datetime) -> tuple[int, int]:
    return dt.year, (dt.month - 1) // 3 + 1


def _aggregate_quarters(daily_candles: list[Candle]) -> list[_AggPeriod]:
    if not daily_candles:
        return []
    result: list[_AggPeriod] = []
    cur_q = _quarter_of(daily_candles[0].open_time)
    start = daily_candles[0].open_time
    q_open = daily_candles[0].open
    q_high = daily_candles[0].high
    q_low = daily_candles[0].low
    for c in daily_candles[1:]:
        q = _quarter_of(c.open_time)
        if q != cur_q:
            result.append(_AggPeriod(start=start, open=q_open, high=q_high, low=q_low))
            cur_q = q
            start = c.open_time
            q_open = c.open
            q_high = c.high
            q_low = c.low
        else:
            q_high = max(q_high, c.high)
            q_low = min(q_low, c.low)
    result.append(_AggPeriod(start=start, open=q_open, high=q_high, low=q_low))
    return result


def _quarterly_levels(
    quarters: list[_AggPeriod],
    t: datetime,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not quarters:
        return None, None, None, None
    idx = bisect_right(quarters, t, key=lambda q: q.start) - 1
    if idx < 0:
        return None, None, None, None
    if idx == 0:
        return quarters[0].open, None, None, None
    prev = quarters[idx - 1]
    eq = (prev.high + prev.low) / 2
    return quarters[idx].open, prev.high, prev.low, eq


# ---------------------------------------------------------------------------
# Helpers — yearly (running cumulative high/low)
# ---------------------------------------------------------------------------

def _build_yearly_running(daily_candles: list[Candle]) -> dict[int, _YearlyRunning]:
    by_year: dict[int, list[Candle]] = {}
    for c in daily_candles:
        by_year.setdefault(c.open_time.year, []).append(c)
    result: dict[int, _YearlyRunning] = {}
    for year, candles in by_year.items():
        cum_highs: list[float] = []
        cum_lows: list[float] = []
        running_high = float("-inf")
        running_low = float("inf")
        for c in candles:
            running_high = max(running_high, c.high)
            running_low = min(running_low, c.low)
            cum_highs.append(running_high)
            cum_lows.append(running_low)
        result[year] = _YearlyRunning(
            year_open=candles[0].open,
            candle_close_times=[c.close_time for c in candles],
            cum_highs=cum_highs,
            cum_lows=cum_lows,
        )
    return result


def _yearly_levels(
    yearly_data: dict[int, _YearlyRunning],
    t: datetime,
) -> tuple[float | None, float | None, float | None, float | None]:
    data = yearly_data.get(t.year)
    if data is None:
        return None, None, None, None
    # Only include completed daily candles (close_time <= t).
    idx = bisect_right(data.candle_close_times, t) - 1
    if idx < 0:
        return data.year_open, None, None, None
    yh = data.cum_highs[idx]
    yl = data.cum_lows[idx]
    return data.year_open, yh, yl, (yh + yl) / 2


# ---------------------------------------------------------------------------
# Helpers — sessions (Asia / London / New York)
# ---------------------------------------------------------------------------

def _build_completed_sessions(
    hourly_candles: list[Candle],
    start_hour: int,
    end_hour: int,
) -> list[_CompletedSession]:
    """Build a sorted list of completed sessions from hourly candles."""
    by_date: dict[object, list[Candle]] = {}  # date -> candles
    for c in hourly_candles:
        if start_hour <= c.open_time.hour < end_hour:
            d = c.open_time.date()
            by_date.setdefault(d, []).append(c)

    expected = end_hour - start_hour
    result: list[_CompletedSession] = []
    for d in sorted(by_date):
        candles = sorted(by_date[d], key=lambda c: c.open_time)
        if len(candles) < expected:
            continue
        result.append(_CompletedSession(
            end_time=candles[-1].close_time,
            open=candles[0].open,
            high=max(c.high for c in candles),
            low=min(c.low for c in candles),
        ))
    return result


# ---------------------------------------------------------------------------
# Helpers — Monday range
# ---------------------------------------------------------------------------

def _build_completed_mondays(hourly_candles: list[Candle]) -> list[_CompletedMonday]:
    """Build a sorted list of completed Monday ranges from hourly candles."""
    by_monday: dict[object, list[Candle]] = {}  # date -> candles
    for c in hourly_candles:
        if c.open_time.weekday() == 0:  # Monday
            d = c.open_time.date()
            by_monday.setdefault(d, []).append(c)

    result: list[_CompletedMonday] = []
    for d in sorted(by_monday):
        candles = by_monday[d]
        if len(candles) < 24:
            continue
        candles_sorted = sorted(candles, key=lambda c: c.open_time)
        high = max(c.high for c in candles)
        low = min(c.low for c in candles)
        result.append(_CompletedMonday(
            end_time=candles_sorted[-1].close_time,
            high=high,
            low=low,
            mid=(high + low) / 2,
        ))
    return result


# ---------------------------------------------------------------------------
# Generic "most recent completed" lookup
# ---------------------------------------------------------------------------

def _latest_before(items: list, t: datetime):
    """Return the most recent item whose ``end_time <= t``, or ``None``."""
    if not items:
        return None
    idx = bisect_right(items, t, key=lambda x: x.end_time) - 1
    if idx < 0:
        return None
    return items[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_key_levels_series(
    h4_candles: list[Candle],
    daily_candles: list[Candle],
    weekly_candles: list[Candle],
    monthly_candles: list[Candle],
    hourly_candles: list[Candle],
    timestamps: list[datetime],
) -> list[KeyLevels]:
    """Compute structural key levels for each timestamp.

    All candle lists must be sorted by ``open_time``.
    ``timestamps`` must be sorted ascending.

    Key levels are lookahead-free by construction: every value is derived
    from completed candles / periods whose close_time <= the query timestamp.
    """
    if not timestamps:
        return []

    # Pre-compute derived structures once.
    quarters = _aggregate_quarters(daily_candles)
    yearly = _build_yearly_running(daily_candles)
    mondays = _build_completed_mondays(hourly_candles)
    asia_sessions = _build_completed_sessions(hourly_candles, ASIA_START_HOUR, ASIA_END_HOUR)
    london_sessions = _build_completed_sessions(hourly_candles, LONDON_START_HOUR, LONDON_END_HOUR)
    ny_sessions = _build_completed_sessions(hourly_candles, NY_START_HOUR, NY_END_HOUR)

    result: list[KeyLevels] = []
    for t in timestamps:
        h4 = _open_prev_levels(h4_candles, t)
        dl = _open_prev_levels(daily_candles, t)
        wk = _open_prev_levels(weekly_candles, t)
        mo = _open_prev_levels(monthly_candles, t)
        qt = _quarterly_levels(quarters, t)
        yr = _yearly_levels(yearly, t)

        mon = _latest_before(mondays, t)
        asia = _latest_before(asia_sessions, t)
        ldn = _latest_before(london_sessions, t)
        ny = _latest_before(ny_sessions, t)

        result.append(KeyLevels(
            h4_open=h4[0], prev_h4_high=h4[1], prev_h4_low=h4[2], h4_eq=h4[3],
            daily_open=dl[0], pdh=dl[1], pdl=dl[2], daily_eq=dl[3],
            weekly_open=wk[0], prev_week_high=wk[1], prev_week_low=wk[2], weekly_eq=wk[3],
            monthly_open=mo[0], prev_month_high=mo[1], prev_month_low=mo[2], monthly_eq=mo[3],
            quarterly_open=qt[0], prev_quarter_high=qt[1], prev_quarter_low=qt[2], quarterly_eq=qt[3],
            yearly_open=yr[0], yearly_high=yr[1], yearly_low=yr[2], yearly_eq=yr[3],
            monday_high=mon.high if mon else None,
            monday_low=mon.low if mon else None,
            monday_mid=mon.mid if mon else None,
            asia_open=asia.open if asia else None,
            asia_high=asia.high if asia else None,
            asia_low=asia.low if asia else None,
            london_open=ldn.open if ldn else None,
            london_high=ldn.high if ldn else None,
            london_low=ldn.low if ldn else None,
            ny_open=ny.open if ny else None,
            ny_high=ny.high if ny else None,
            ny_low=ny.low if ny else None,
        ))

    return result
