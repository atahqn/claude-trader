from __future__ import annotations

import random as _random_module
from datetime import datetime, timedelta
from typing import Any

from .data import BinanceClient
from .models import (
    BacktestResult,
    ExitReason,
    PositionType,
    ResolutionLevel,
    Signal,
    TradeResult,
)
from .pipeline import BacktestExecutionSession, PreparedMarketContext
from .resolver import compute_pnl, compute_tp_sl_prices, resolve_exit


_APPROXIMATE_ENTRY_INTERVALS: tuple[tuple[str, timedelta], ...] = (
    ("1h", timedelta(hours=1)),
    ("30m", timedelta(minutes=30)),
    ("15m", timedelta(minutes=15)),
    ("5m", timedelta(minutes=5)),
    ("1m", timedelta(minutes=1)),
)

DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS = 15


def _resolved_entry_delay_seconds(
    signal: Signal,
    default_entry_delay_seconds: int = DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS,
) -> int:
    if signal.entry_delay_seconds is None:
        return default_entry_delay_seconds
    return signal.entry_delay_seconds



def _resolve_timeout_exit(
    signal: Signal,
    session: BacktestExecutionSession,
    timeout_time: datetime,
    entry_price: float,
    *,
    approximate: bool = False,
) -> tuple[float, datetime, ResolutionLevel, bool]:
    """Approximate a live timeout close using the first trade after the deadline."""
    ticker = signal.ticker

    if not approximate:
        timeout_trades = session.fetch_agg_trades(
            ticker,
            timeout_time,
            timeout_time + timedelta(minutes=5),
        )
        first_trade = next((t for t in timeout_trades if t.timestamp >= timeout_time), None)
        if first_trade is not None:
            return first_trade.price, first_trade.timestamp, ResolutionLevel.TRADE, False

    minute_start = timeout_time.replace(second=0, microsecond=0)
    minute_candles = session.fetch_minute_candles(
        ticker,
        minute_start,
        minute_start + timedelta(minutes=2),
    )
    minute_candle = next((c for c in minute_candles if c.close_time >= timeout_time), None)
    if minute_candle is not None:
        return (
            minute_candle.close,
            minute_candle.close_time,
            ResolutionLevel.MINUTE,
            not approximate,
        )

    hour_start = timeout_time.replace(minute=0, second=0, microsecond=0)
    hour_candles = session.fetch_hourly_candles(
        ticker,
        hour_start,
        hour_start + timedelta(hours=2),
    )
    hour_candle = next((c for c in hour_candles if c.close_time >= timeout_time), None)
    if hour_candle is not None:
        return (
            hour_candle.close,
            hour_candle.close_time,
            ResolutionLevel.HOUR,
            not approximate,
        )

    fallback_level = ResolutionLevel.HOUR if approximate else ResolutionLevel.TRADE
    return entry_price, timeout_time, fallback_level, not approximate


def _resolve_entry_approximate(
    signal: Signal,
    session: BacktestExecutionSession,
    *,
    default_entry_delay_seconds: int = DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS,
) -> tuple[float, datetime] | None:
    """Approximate entry using the last fully closed aligned candle price."""
    ticker = signal.ticker
    delay = timedelta(
        seconds=_resolved_entry_delay_seconds(
            signal,
            default_entry_delay_seconds=default_entry_delay_seconds,
        )
    )
    candle_interval, candle_close_time = _select_approximate_entry_candle(signal.signal_date)
    candle_start = candle_close_time - _interval_duration(candle_interval)

    if candle_interval == "1h":
        candles = session.fetch_hourly_candles(ticker, candle_start, candle_close_time)
        candle = next((c for c in reversed(candles) if c.close_time <= candle_close_time), None)
    else:
        minute_candles = session.fetch_minute_candles(ticker, candle_start, candle_close_time)
        candle = _aggregate_entry_candle(minute_candles, candle_start, candle_close_time)

    if candle is None:
        return None

    entry_time = signal.signal_date + delay
    return candle.close, entry_time


def _select_approximate_entry_candle(signal_time: datetime) -> tuple[str, datetime]:
    candle_close_time = signal_time.replace(second=0, microsecond=0)
    minute = candle_close_time.minute
    for interval, duration in _APPROXIMATE_ENTRY_INTERVALS:
        duration_minutes = int(duration.total_seconds() // 60)
        if duration_minutes == 60:
            if minute == 0:
                return interval, candle_close_time
            continue
        if minute % duration_minutes == 0:
            return interval, candle_close_time
    return "1m", candle_close_time


def _interval_duration(interval: str) -> timedelta:
    for name, duration in _APPROXIMATE_ENTRY_INTERVALS:
        if name == interval:
            return duration
    raise ValueError(f"unsupported interval: {interval}")


def _aggregate_entry_candle(
    candles: list[Any],
    open_time: datetime,
    close_time: datetime,
) -> Any | None:
    if not candles:
        return None

    first = candles[0]
    last = candles[-1]
    return type(first)(
        open_time=open_time,
        close_time=close_time,
        open=first.open,
        high=max(candle.high for candle in candles),
        low=min(candle.low for candle in candles),
        close=last.close,
        volume=sum(candle.volume for candle in candles),
    )


def _ensure_session(
    *,
    signal: Signal | None = None,
    client: BinanceClient | None = None,
    session: BacktestExecutionSession | None = None,
    prepared_context: PreparedMarketContext | None = None,
    use_chunk_cache: bool = True,
) -> BacktestExecutionSession:
    if session is not None:
        return session

    if client is None:
        market_type = signal.market_type if signal is not None else None
        if market_type is None:
            client = BinanceClient()
        else:
            client = BinanceClient(market_type=market_type)

    return BacktestExecutionSession(
        client=client,
        prepared_context=prepared_context,
        use_chunk_cache=use_chunk_cache,
    )


def backtest_signal(
    signal: Signal,
    client: BinanceClient | None = None,
    approximate: bool = False,
    rng: _random_module.Random | None = None,
    session: BacktestExecutionSession | None = None,
    prepared_context: PreparedMarketContext | None = None,
    default_entry_delay_seconds: int = DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS,
) -> TradeResult:
    """Backtest a single signal and return its trade result."""
    session = _ensure_session(
        signal=signal,
        client=client,
        session=session,
        prepared_context=prepared_context,
        use_chunk_cache=False,
    )

    ticker = signal.ticker
    if rng is None:
        rng = _random_module.Random()
    used_fallback = False

    # ----- Step 1: Resolve entry (fill check) -----
    if approximate:
        approx_entry = _resolve_entry_approximate(
            signal,
            session,
            default_entry_delay_seconds=default_entry_delay_seconds,
        )
        if approx_entry is None:
            return _unfilled_result(signal)
        entry_price, entry_time = approx_entry
    elif signal.entry_price is None:
        # Market order: entry = first trade after signal_date + entry_delay
        delay = timedelta(
            seconds=_resolved_entry_delay_seconds(
                signal,
                default_entry_delay_seconds=default_entry_delay_seconds,
            )
        )
        window_end = signal.signal_date + delay + timedelta(seconds=10)
        trades = session.fetch_agg_trades(ticker, signal.signal_date, window_end)
        earliest = signal.signal_date + delay
        base_trade = next((t for t in trades if t.timestamp >= earliest), None)
        if base_trade is not None:
            entry_price = base_trade.price
            entry_time = base_trade.timestamp
        else:
            # Fallback: close of 1m candle at signal_date
            candle_start = signal.signal_date.replace(second=0, microsecond=0)
            candle_end = candle_start + timedelta(minutes=1)
            fallback_candles = session.fetch_minute_candles(ticker, candle_start, candle_end)
            if fallback_candles:
                entry_price = fallback_candles[0].close
                entry_time = fallback_candles[0].close_time
                used_fallback = True
            else:
                # Cannot determine entry at all -> treat as unfilled
                return _unfilled_result(signal)
    else:
        # Limit order: scan aggTrades for fill
        fill_end = signal.signal_date + timedelta(seconds=signal.fill_timeout_seconds)
        trades = session.fetch_agg_trades(ticker, signal.signal_date, fill_end)
        fill_trade = None
        for t in trades:
            if signal.position_type is PositionType.LONG:
                if t.price <= signal.entry_price:
                    fill_trade = t
                    break
            else:
                if t.price >= signal.entry_price:
                    fill_trade = t
                    break
        if fill_trade is None:
            return _unfilled_result(signal)
        entry_price = signal.entry_price
        entry_time = fill_trade.timestamp

    # ----- Step 2: Compute TP/SL prices -----
    tp_price, sl_price = compute_tp_sl_prices(
        entry_price,
        signal.position_type,
        tp_pct=signal.tp_pct,
        sl_pct=signal.sl_pct,
        taker_fee_rate=signal.taker_fee_rate,
        tp_price_override=signal.tp_price,
        sl_price_override=signal.sl_price,
    )

    # ----- Step 3: Fetch 1h candles for resolution window -----
    holding_hours = signal.max_holding_hours
    resolution_end = entry_time + timedelta(hours=holding_hours)
    hour_candles = session.fetch_hourly_candles(ticker, entry_time, resolution_end)

    # ----- Step 4: Resolve exit (3-level hierarchy) -----
    def minute_fetcher(start, end):
        return session.fetch_minute_candles(ticker, start, end)

    def agg_trade_fetcher(start, end):
        return session.fetch_agg_trades(ticker, start, end)

    resolution = resolve_exit(
        hour_candles,
        signal.position_type,
        tp_price,
        sl_price,
        entry_time,
        minute_fetcher,
        agg_trade_fetcher,
        end_time=resolution_end,
        approximate=approximate,
        rng=rng,
    )

    # ----- Step 5: Build result -----
    if resolution is not None:
        exit_price = resolution.exit_price
        exit_time = resolution.exit_time
        exit_reason = resolution.reason
        resolution_level = resolution.resolution_level
        used_fallback = used_fallback or resolution.used_fallback
    else:
        # Timeout: no exit within max_hours
        exit_reason = ExitReason.TIMEOUT
        exit_price, exit_time, resolution_level, timeout_used_fallback = _resolve_timeout_exit(
            signal,
            session,
            resolution_end,
            entry_price,
            approximate=approximate,
        )
        used_fallback = used_fallback or timeout_used_fallback

    net_pnl, gross_pnl, fee_drag = compute_pnl(
        entry_price, exit_price, signal.position_type,
        signal.leverage, signal.taker_fee_rate,
    )

    return TradeResult(
        signal=signal,
        entry_price=entry_price,
        entry_time=entry_time,
        exit_price=exit_price,
        exit_time=exit_time,
        exit_reason=exit_reason,
        resolution_level=resolution_level,
        tp_price=tp_price,
        sl_price=sl_price,
        pnl_pct=net_pnl,
        gross_pnl_pct=gross_pnl,
        fee_drag_pct=fee_drag,
        used_fallback=used_fallback,
    )


def _compute_stats(trades: list[TradeResult]) -> dict:
    """Compute aggregate stats from a sorted trade list."""
    wins = sum(1 for t in trades if t.exit_reason is not ExitReason.UNFILLED and t.pnl_pct > 0)
    losses = sum(1 for t in trades if t.exit_reason is not ExitReason.UNFILLED and t.pnl_pct <= 0)
    unfilled = sum(1 for t in trades if t.exit_reason is ExitReason.UNFILLED)
    open_trades = sum(1 for t in trades if t.exit_reason is ExitReason.TIMEOUT)
    resolved = [t for t in trades if t.exit_reason is not ExitReason.UNFILLED]
    total = len(resolved)

    total_pnl = sum(t.pnl_pct * t.signal.size_multiplier for t in resolved)
    avg_pnl = total_pnl / total if total > 0 else 0.0
    win_rate = (wins / total * 100) if total > 0 else 0.0

    gross_wins = sum(t.pnl_pct * t.signal.size_multiplier for t in resolved if t.pnl_pct > 0)
    gross_losses = abs(sum(t.pnl_pct * t.signal.size_multiplier for t in resolved if t.pnl_pct <= 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    equity_curve = [equity]
    for t in resolved:
        equity *= (1 + t.pnl_pct * t.signal.size_multiplier / 100)
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "open_trades": open_trades,
        "unfilled": unfilled,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl,
        "avg_pnl_pct": avg_pnl,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "equity_curve": equity_curve,
    }


def backtest_signals(
    signals: list[Signal],
    client: BinanceClient | None = None,
    approximate: bool = False,
    seed: int | None = None,
    session: BacktestExecutionSession | None = None,
    prepared_context: PreparedMarketContext | None = None,
    default_entry_delay_seconds: int = DEFAULT_BACKTEST_ENTRY_DELAY_SECONDS,
) -> BacktestResult:
    """Backtest multiple signals and aggregate results."""
    rng = _random_module.Random(seed)
    session = _ensure_session(
        signal=signals[0] if signals else None,
        client=client,
        session=session,
        prepared_context=prepared_context,
        use_chunk_cache=True,
    )
    trades: list[TradeResult] = []
    for signal in signals:
        result = backtest_signal(
            signal,
            approximate=approximate,
            rng=rng,
            session=session,
            default_entry_delay_seconds=default_entry_delay_seconds,
        )
        trades.append(result)

    trades.sort(key=lambda t: t.entry_time)
    stats = _compute_stats(trades)
    return BacktestResult(trades=trades, **stats)


def _unfilled_result(signal: Signal) -> TradeResult:
    return TradeResult(
        signal=signal,
        entry_price=0.0,
        entry_time=signal.signal_date,
        exit_price=0.0,
        exit_time=signal.signal_date,
        exit_reason=ExitReason.UNFILLED,
        resolution_level=ResolutionLevel.HOUR,
        tp_price=0.0,
        sl_price=0.0,
        pnl_pct=0.0,
        gross_pnl_pct=0.0,
        fee_drag_pct=0.0,
        used_fallback=False,
    )
