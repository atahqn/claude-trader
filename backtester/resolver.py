from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from .models import (
    AggTrade,
    Candle,
    ExitReason,
    ExitResolution,
    PositionType,
    ResolutionLevel,
)


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


def resolve_exit(
    hour_candles: list[Candle],
    position_type: PositionType,
    tp_price: float,
    sl_price: float,
    entry_time: datetime,
    minute_fetcher: MinuteFetcher,
    agg_trade_fetcher: AggTradeFetcher,
) -> ExitResolution | None:
    """3-level hierarchical exit resolution: HOUR -> MINUTE -> TRADE.

    Ported from kriptistan execution.py resolve_exit_hierarchical.
    """
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

    # --- Subsequent hours ---
    for candle in hour_candles:
        if candle.open_time < first_hour_end:
            continue
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
    return ExitResolution(ExitReason.SL, close_time, sl_price, ResolutionLevel.HOUR)


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
        return ExitResolution(ExitReason.SL, candle.close_time, sl_price, ResolutionLevel.MINUTE)
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
