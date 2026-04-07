from __future__ import annotations

import random as _random_module
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from .models import (
    AggTrade,
    Candle,
    ExitReason,
    ExitResolution,
    PositionType,
    ResolutionLevel,
)

import os as _os

try:
    from resolver_rs import resolve_exit_rs as _rs_resolve_exit
    _RUST_AVAILABLE = not _os.environ.get("RESOLVER_NO_RUST")
except ImportError:
    _RUST_AVAILABLE = False


def compute_tp_sl_prices(
    entry_price: float,
    position_type: PositionType,
    *,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
    taker_fee_rate: float = 0.0005,
    tp_price_override: float | None = None,
    sl_price_override: float | None = None,
) -> tuple[float, float]:
    """Compute TP and SL price levels.

    If explicit prices are given they are used directly (no fee adjustment).
    Otherwise prices are derived from percentages with fee offset baked in.
    """
    if tp_price_override is not None:
        tp_price = tp_price_override
    else:
        if tp_pct is None:
            raise ValueError("tp_pct required when tp_price_override is not set")
        fee_offset = taker_fee_rate * 2 * 100
        tp_with_fees = tp_pct + fee_offset
        if position_type is PositionType.LONG:
            tp_price = entry_price * (1 + tp_with_fees / 100)
        else:
            tp_price = entry_price * (1 - tp_with_fees / 100)

    if sl_price_override is not None:
        sl_price = sl_price_override
    else:
        if sl_pct is None:
            raise ValueError("sl_pct required when sl_price_override is not set")
        fee_offset = taker_fee_rate * 2 * 100
        sl_with_fees = max(sl_pct - fee_offset, 0.01)
        if position_type is PositionType.LONG:
            sl_price = entry_price * (1 - sl_with_fees / 100)
        else:
            sl_price = entry_price * (1 + sl_with_fees / 100)

    return tp_price, sl_price


def compute_pnl(
    entry_price: float,
    exit_price: float,
    position_type: PositionType,
    leverage: float = 1.0,
    taker_fee_rate: float = 0.0005,
) -> tuple[float, float, float]:
    """Return (net_pnl_pct, gross_pnl_pct, fee_drag_pct)."""
    if position_type is PositionType.LONG:
        gross = ((exit_price - entry_price) / entry_price) * 100 * leverage
    else:
        gross = ((entry_price - exit_price) / entry_price) * 100 * leverage
    fee_drag = 2 * taker_fee_rate * leverage * 100
    return gross - fee_drag, gross, fee_drag


def barrier_outcome(
    candle: Candle,
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
) -> ExitReason | None:
    """Check a single candle against TP/SL barriers.

    Returns TP, SL, or None (ambiguous — both hit).
    Returns ExitReason with a sentinel value via the TIMEOUT member if neither hit
    (we re-use TIMEOUT as 'OPEN' internally, but callers should treat None as ambiguous).
    """
    if position_type is PositionType.LONG:
        tp_touched = candle.high >= tp_price
        sl_touched = candle.low <= sl_price
    else:
        tp_touched = candle.low <= tp_price
        sl_touched = candle.high >= sl_price
    if not tp_touched and not sl_touched:
        return ExitReason.TIMEOUT  # sentinel: neither barrier hit
    if tp_touched and sl_touched:
        return None  # ambiguous
    return ExitReason.TP if tp_touched else ExitReason.SL


def resolve_with_agg_trades(
    trades: list[AggTrade],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    start_time: datetime,
) -> ExitResolution | None:
    """Walk trades chronologically, return first barrier hit."""
    for trade in trades:
        if trade.timestamp < start_time:
            continue
        if position_type is PositionType.LONG:
            if trade.price >= tp_price:
                return ExitResolution(ExitReason.TP, trade.timestamp, tp_price, ResolutionLevel.TRADE)
            if trade.price <= sl_price:
                return ExitResolution(ExitReason.SL, trade.timestamp, sl_price, ResolutionLevel.TRADE)
        else:
            if trade.price <= tp_price:
                return ExitResolution(ExitReason.TP, trade.timestamp, tp_price, ResolutionLevel.TRADE)
            if trade.price >= sl_price:
                return ExitResolution(ExitReason.SL, trade.timestamp, sl_price, ResolutionLevel.TRADE)
    return None


# Type aliases for fetcher callbacks
MinuteFetcher = Callable[[datetime, datetime], list[Candle]]
AggTradeFetcher = Callable[[datetime, datetime], list[AggTrade]]
ApproximationLogger = Callable[[str], None]


def _dt_to_ms(dt: datetime) -> int:
    """Convert datetime to UTC milliseconds since epoch."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_dt(ms: int) -> datetime:
    """Convert UTC milliseconds to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _resolve_exit_py(
    hour_candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    entry_time: datetime,
    minute_fetcher: MinuteFetcher,
    agg_trade_fetcher: AggTradeFetcher,
    end_time: datetime | None = None,
    approximate: bool = False,
    rng: _random_module.Random | None = None,
    logger: ApproximationLogger | None = None,
) -> ExitResolution | None:
    """3-level hierarchical exit resolution: HOUR -> MINUTE -> TRADE.

    Ported from kriptistan execution.py resolve_exit_hierarchical.
    """
    if approximate:
        return resolve_exit_approximate(
            hour_candles,
            position_type,
            tp_price,
            sl_price,
            entry_time,
            minute_fetcher,
            end_time=end_time,
            rng=rng,
            logger=logger,
        )

    # --- First hour: entry may be mid-hour, protect partial minute with trades ---
    first_hour_end = entry_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Resolve the entry minute with exact trades
    entry_minute_start = entry_time.replace(second=0, microsecond=0)
    entry_minute_end = entry_minute_start + timedelta(minutes=1)
    entry_trades = agg_trade_fetcher(entry_time, entry_minute_end)
    trade_hit = resolve_with_agg_trades(entry_trades, position_type, tp_price, sl_price, entry_time)
    if trade_hit is not None:
        return trade_hit

    # Remaining minutes in the first hour
    same_hour_minutes = minute_fetcher(entry_minute_end, first_hour_end)
    first_hour_result = _resolve_hour_interval(
        same_hour_minutes, position_type, tp_price, sl_price, first_hour_end, agg_trade_fetcher,
    )
    if first_hour_result is not None:
        return first_hour_result

    if end_time is not None and end_time <= first_hour_end:
        return None

    # --- Subsequent hours ---
    final_hour_start = (
        end_time.replace(minute=0, second=0, microsecond=0)
        if end_time is not None else None
    )
    for candle in hour_candles:
        if candle.open_time < first_hour_end:
            continue
        if final_hour_start is not None and candle.open_time >= final_hour_start:
            break
        outcome = barrier_outcome(candle, position_type, tp_price, sl_price)
        if outcome == ExitReason.TIMEOUT:
            # Neither barrier hit this hour
            continue
        if outcome in (ExitReason.TP, ExitReason.SL):
            # Unambiguous at hour level
            exit_price = tp_price if outcome is ExitReason.TP else sl_price
            return ExitResolution(outcome, candle.close_time, exit_price, ResolutionLevel.HOUR)
        # Ambiguous -> drill into minutes
        hour_minutes = minute_fetcher(candle.open_time, candle.close_time)
        nested = _resolve_candles_minute(
            hour_minutes, position_type, tp_price, sl_price, agg_trade_fetcher,
        )
        if nested is not None:
            return nested

    if end_time is not None and final_hour_start is not None and end_time > final_hour_start:
        trailing = _resolve_partial_interval(
            final_hour_start,
            end_time,
            position_type,
            tp_price,
            sl_price,
            minute_fetcher,
            agg_trade_fetcher,
        )
        if trailing is not None:
            return trailing

    return None


def resolve_exit_approximate(
    hour_candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    entry_time: datetime,
    minute_fetcher: MinuteFetcher,
    end_time: datetime | None = None,
    rng: _random_module.Random | None = None,
    logger: ApproximationLogger | None = None,
) -> ExitResolution | None:
    """Approximate exit resolution using hour/minute candles only.

    Ambiguous minute candles are resolved randomly because aggTrades are disabled.
    """
    if rng is None:
        rng = _random_module.Random()
    if logger is None:
        logger = print

    first_hour_end = entry_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    entry_minute_start = entry_time.replace(second=0, microsecond=0)

    same_hour_minutes = minute_fetcher(entry_minute_start, first_hour_end)
    first_hour_result = _resolve_hour_interval_approximate(
        same_hour_minutes,
        position_type,
        tp_price,
        sl_price,
        first_hour_end,
        rng,
        logger,
    )
    if first_hour_result is not None:
        return first_hour_result

    if end_time is not None and end_time <= first_hour_end:
        return None

    final_hour_start = (
        end_time.replace(minute=0, second=0, microsecond=0)
        if end_time is not None else None
    )
    for candle in hour_candles:
        if candle.open_time < first_hour_end:
            continue
        if final_hour_start is not None and candle.open_time >= final_hour_start:
            break
        outcome = barrier_outcome(candle, position_type, tp_price, sl_price)
        if outcome == ExitReason.TIMEOUT:
            continue
        if outcome in (ExitReason.TP, ExitReason.SL):
            exit_price = tp_price if outcome is ExitReason.TP else sl_price
            return ExitResolution(outcome, candle.close_time, exit_price, ResolutionLevel.HOUR)

        hour_minutes = minute_fetcher(candle.open_time, candle.close_time)
        nested = _resolve_candles_minute_approximate(
            hour_minutes,
            position_type,
            tp_price,
            sl_price,
            rng,
            logger,
        )
        if nested is not None:
            return nested

    if end_time is not None and final_hour_start is not None and end_time > final_hour_start:
        trailing = _resolve_partial_interval_approximate(
            final_hour_start,
            end_time,
            position_type,
            tp_price,
            sl_price,
            minute_fetcher,
            rng,
            logger,
        )
        if trailing is not None:
            return trailing

    return None


def _resolve_hour_interval(
    candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    close_time: datetime,
    agg_trade_fetcher: AggTradeFetcher,
) -> ExitResolution | None:
    """Resolve an interval of minute candles within a single hour."""
    outcome = _barrier_outcome_for_candles(candles, position_type, tp_price, sl_price)
    if outcome == ExitReason.TIMEOUT:
        return None
    if outcome in (ExitReason.TP, ExitReason.SL):
        exit_price = tp_price if outcome is ExitReason.TP else sl_price
        return ExitResolution(outcome, close_time, exit_price, ResolutionLevel.HOUR)
    # Ambiguous -> drill into minute-by-minute
    nested = _resolve_candles_minute(candles, position_type, tp_price, sl_price, agg_trade_fetcher)
    if nested is not None:
        return nested
    # Fallback: SL at hour close (matching kriptistan behavior)
    return ExitResolution(
        ExitReason.SL,
        close_time,
        sl_price,
        ResolutionLevel.HOUR,
        used_fallback=True,
    )


def _resolve_candles_minute(
    candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    agg_trade_fetcher: AggTradeFetcher,
) -> ExitResolution | None:
    """Walk minute candles, drill into aggTrades on ambiguity."""
    for candle in candles:
        outcome = barrier_outcome(candle, position_type, tp_price, sl_price)
        if outcome == ExitReason.TIMEOUT:
            continue
        if outcome in (ExitReason.TP, ExitReason.SL):
            exit_price = tp_price if outcome is ExitReason.TP else sl_price
            return ExitResolution(outcome, candle.close_time, exit_price, ResolutionLevel.MINUTE)
        # Ambiguous at minute -> drill into trades
        trades = agg_trade_fetcher(candle.open_time, candle.close_time)
        nested = resolve_with_agg_trades(trades, position_type, tp_price, sl_price, candle.open_time)
        if nested is not None:
            return nested
        # Fallback: SL at minute close
        return ExitResolution(
            ExitReason.SL,
            candle.close_time,
            sl_price,
            ResolutionLevel.MINUTE,
            used_fallback=True,
        )
    return None


def _resolve_hour_interval_approximate(
    candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    close_time: datetime,
    rng: _random_module.Random,
    logger: ApproximationLogger,
) -> ExitResolution | None:
    """Resolve an interval of minute candles without trade-level data."""
    outcome = _barrier_outcome_for_candles(candles, position_type, tp_price, sl_price)
    if outcome == ExitReason.TIMEOUT:
        return None
    if outcome in (ExitReason.TP, ExitReason.SL):
        exit_price = tp_price if outcome is ExitReason.TP else sl_price
        return ExitResolution(outcome, close_time, exit_price, ResolutionLevel.HOUR)
    nested = _resolve_candles_minute_approximate(
        candles,
        position_type,
        tp_price,
        sl_price,
        rng,
        logger,
    )
    if nested is not None:
        return nested
    return ExitResolution(ExitReason.SL, close_time, sl_price, ResolutionLevel.HOUR)


def _resolve_candles_minute_approximate(
    candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    rng: _random_module.Random,
    logger: ApproximationLogger,
) -> ExitResolution | None:
    """Walk minute candles and randomize when the minute itself is ambiguous."""
    for candle in candles:
        outcome = barrier_outcome(candle, position_type, tp_price, sl_price)
        if outcome == ExitReason.TIMEOUT:
            continue
        if outcome in (ExitReason.TP, ExitReason.SL):
            exit_price = tp_price if outcome is ExitReason.TP else sl_price
            return ExitResolution(outcome, candle.close_time, exit_price, ResolutionLevel.MINUTE)

        random_outcome = rng.choice([ExitReason.TP, ExitReason.SL])
        logger(
            "[approximate] cannot resolve TP/SL within 1 minute without aggTrades; "
            f"picked {random_outcome.value} randomly for candle ending {candle.close_time.isoformat()}"
        )
        exit_price = tp_price if random_outcome is ExitReason.TP else sl_price
        return ExitResolution(random_outcome, candle.close_time, exit_price, ResolutionLevel.MINUTE)
    return None


def _resolve_partial_interval(
    start_time: datetime,
    end_time: datetime,
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    minute_fetcher: MinuteFetcher,
    agg_trade_fetcher: AggTradeFetcher,
) -> ExitResolution | None:
    """Resolve a trailing partial-hour interval without looking past *end_time*."""
    if start_time >= end_time:
        return None

    full_minute_end = end_time.replace(second=0, microsecond=0)
    if start_time < full_minute_end:
        minutes = minute_fetcher(start_time, full_minute_end)
        nested = _resolve_candles_minute(
            minutes, position_type, tp_price, sl_price, agg_trade_fetcher,
        )
        if nested is not None:
            return nested

    if full_minute_end < end_time:
        trades = agg_trade_fetcher(full_minute_end, end_time)
        return resolve_with_agg_trades(
            trades, position_type, tp_price, sl_price, full_minute_end,
        )

    return None


def _resolve_partial_interval_approximate(
    start_time: datetime,
    end_time: datetime,
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    minute_fetcher: MinuteFetcher,
    rng: _random_module.Random,
    logger: ApproximationLogger,
) -> ExitResolution | None:
    """Resolve a trailing partial-hour interval without aggTrades."""
    if start_time >= end_time:
        return None

    full_minute_end = end_time.replace(second=0, microsecond=0)
    if start_time < full_minute_end:
        minutes = minute_fetcher(start_time, full_minute_end)
        nested = _resolve_candles_minute_approximate(
            minutes,
            position_type,
            tp_price,
            sl_price,
            rng,
            logger,
        )
        if nested is not None:
            return nested

    return None


def _barrier_outcome_for_candles(
    candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
) -> ExitReason | None:
    """Aggregate barrier check across multiple candles."""
    if not candles:
        return ExitReason.TIMEOUT
    tp_touched = False
    sl_touched = False
    for candle in candles:
        if position_type is PositionType.LONG:
            tp_touched = tp_touched or candle.high >= tp_price
            sl_touched = sl_touched or candle.low <= sl_price
        else:
            tp_touched = tp_touched or candle.low <= tp_price
            sl_touched = sl_touched or candle.high >= sl_price
        if tp_touched and sl_touched:
            return None
    if not tp_touched and not sl_touched:
        return ExitReason.TIMEOUT
    return ExitReason.TP if tp_touched else ExitReason.SL


def resolve_exit(
    hour_candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    entry_time: datetime,
    minute_fetcher: MinuteFetcher,
    agg_trade_fetcher: AggTradeFetcher,
    end_time: datetime | None = None,
    approximate: bool = False,
    rng: _random_module.Random | None = None,
    logger: ApproximationLogger | None = None,
) -> ExitResolution | None:
    """3-level hierarchical exit resolution: HOUR -> MINUTE -> TRADE.

    Dispatches to the native Rust implementation when available, otherwise
    falls back to the pure-Python resolver.
    """
    if _RUST_AVAILABLE and not approximate:
        candle_tuples = [
            (_dt_to_ms(c.open_time), _dt_to_ms(c.close_time), c.high, c.low)
            for c in hour_candles
        ]

        def _wrap_min(start_ms: int, end_ms: int):
            candles = minute_fetcher(_ms_to_dt(start_ms), _ms_to_dt(end_ms))
            return [
                (_dt_to_ms(c.open_time), _dt_to_ms(c.close_time), c.high, c.low)
                for c in candles
            ]

        def _wrap_agg(start_ms: int, end_ms: int):
            trades = agg_trade_fetcher(_ms_to_dt(start_ms), _ms_to_dt(end_ms))
            return [(_dt_to_ms(t.timestamp), t.price) for t in trades]

        seed = rng.getrandbits(64) if (approximate and rng is not None) else None

        result = _rs_resolve_exit(
            candle_tuples,
            position_type is PositionType.LONG,
            tp_price,
            sl_price,
            _dt_to_ms(entry_time),
            _wrap_min,
            _wrap_agg,
            _dt_to_ms(end_time) if end_time is not None else None,
            approximate,
            seed,
        )

        if result is None:
            return None

        reason_str, time_ms, price, level_str, used_fallback = result
        return ExitResolution(
            reason=ExitReason(reason_str),
            exit_time=_ms_to_dt(time_ms),
            exit_price=price,
            resolution_level=ResolutionLevel(level_str),
            used_fallback=used_fallback,
        )

    return _resolve_exit_py(
        hour_candles,
        position_type,
        tp_price,
        sl_price,
        entry_time,
        minute_fetcher,
        agg_trade_fetcher,
        end_time,
        approximate,
        rng,
        logger,
    )
