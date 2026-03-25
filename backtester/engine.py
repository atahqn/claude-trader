from __future__ import annotations

import random as _random_module
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

from .data import BinanceClient
from .models import (
    BacktestResult,
    ExitReason,
    PortfolioResult,
    PositionType,
    ResolutionLevel,
    Signal,
    SkipReason,
    SkippedSignal,
    TradeResult,
)
from .pipeline import BacktestExecutionSession, PreparedMarketContext
from .resolver import compute_pnl, compute_tp_sl_prices, resolve_exit


def _effective_max_holding_hours(signal: Signal, default_hours: int) -> int:
    holding_hours = signal.max_holding_hours if signal.max_holding_hours is not None else default_hours
    if holding_hours <= 0:
        raise ValueError("max holding hours must be positive")
    return holding_hours


def _resolve_timeout_exit(
    signal: Signal,
    session: BacktestExecutionSession,
    timeout_time: datetime,
    entry_price: float,
    *,
    approximate: bool = False,
) -> tuple[float, datetime, ResolutionLevel]:
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
            return first_trade.price, first_trade.timestamp, ResolutionLevel.TRADE

    minute_start = timeout_time.replace(second=0, microsecond=0)
    minute_candles = session.fetch_minute_candles(
        ticker,
        minute_start,
        minute_start + timedelta(minutes=2),
    )
    minute_candle = next((c for c in minute_candles if c.close_time >= timeout_time), None)
    if minute_candle is not None:
        return minute_candle.close, minute_candle.close_time, ResolutionLevel.MINUTE

    hour_start = timeout_time.replace(minute=0, second=0, microsecond=0)
    hour_candles = session.fetch_hourly_candles(
        ticker,
        hour_start,
        hour_start + timedelta(hours=2),
    )
    hour_candle = next((c for c in hour_candles if c.close_time >= timeout_time), None)
    if hour_candle is not None:
        return hour_candle.close, hour_candle.close_time, ResolutionLevel.HOUR

    fallback_level = ResolutionLevel.HOUR if approximate else ResolutionLevel.TRADE
    return entry_price, timeout_time, fallback_level


def _resolve_entry_approximate(
    signal: Signal,
    session: BacktestExecutionSession,
) -> tuple[float, datetime] | None:
    """Approximate entry using the nearest closed 1h candle price."""
    ticker = signal.ticker
    delay = timedelta(seconds=signal.entry_delay_seconds)

    candle_start = signal.signal_date - timedelta(hours=1)
    candle_end = signal.signal_date + timedelta(hours=1)
    candles = session.fetch_hourly_candles(ticker, candle_start, candle_end)
    candle = next((c for c in candles if c.close_time >= signal.signal_date), None)
    if candle is None:
        candle = candles[-1] if candles else None
    if candle is None:
        return None

    entry_time = signal.signal_date + delay
    return candle.close, entry_time


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
    max_hours: int = 168,
    approximate: bool = False,
    rng: _random_module.Random | None = None,
    session: BacktestExecutionSession | None = None,
    prepared_context: PreparedMarketContext | None = None,
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

    # ----- Step 1: Resolve entry (fill check) -----
    if approximate:
        approx_entry = _resolve_entry_approximate(signal, session)
        if approx_entry is None:
            return _unfilled_result(signal)
        entry_price, entry_time = approx_entry
    elif signal.entry_price is None:
        # Market order: entry = first trade after signal_date + entry_delay
        delay = timedelta(seconds=signal.entry_delay_seconds)
        window_end = signal.signal_date + delay + timedelta(seconds=10)
        trades = session.fetch_agg_trades(ticker, signal.signal_date, window_end)
        earliest = signal.signal_date + delay
        base_trade = next((t for t in trades if t.timestamp >= earliest), None)
        if base_trade is not None:
            entry_price = base_trade.price
            entry_time = base_trade.timestamp
        else:
            # Fallback: close of 1h candle at signal_date
            candle_start = signal.signal_date.replace(minute=0, second=0, microsecond=0)
            candle_end = candle_start + timedelta(hours=1)
            fallback_candles = session.fetch_hourly_candles(ticker, candle_start, candle_end)
            if fallback_candles:
                entry_price = fallback_candles[0].close
                entry_time = fallback_candles[0].close_time
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
    holding_hours = _effective_max_holding_hours(signal, max_hours)
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
    else:
        # Timeout: no exit within max_hours
        exit_reason = ExitReason.TIMEOUT
        exit_price, exit_time, resolution_level = _resolve_timeout_exit(
            signal,
            session,
            resolution_end,
            entry_price,
            approximate=approximate,
        )

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
    )


def _compute_stats(trades: list[TradeResult]) -> dict:
    """Compute aggregate stats from a sorted trade list."""
    wins = sum(1 for t in trades if t.exit_reason is not ExitReason.UNFILLED and t.pnl_pct > 0)
    losses = sum(1 for t in trades if t.exit_reason is not ExitReason.UNFILLED and t.pnl_pct <= 0)
    unfilled = sum(1 for t in trades if t.exit_reason is ExitReason.UNFILLED)
    open_trades = sum(1 for t in trades if t.exit_reason is ExitReason.TIMEOUT)
    resolved = [t for t in trades if t.exit_reason is not ExitReason.UNFILLED]
    total = len(resolved)

    total_pnl = sum(t.pnl_pct for t in resolved)
    avg_pnl = total_pnl / total if total > 0 else 0.0
    win_rate = (wins / total * 100) if total > 0 else 0.0

    gross_wins = sum(t.pnl_pct for t in resolved if t.pnl_pct > 0)
    gross_losses = abs(sum(t.pnl_pct for t in resolved if t.pnl_pct <= 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    equity_curve = [equity]
    for t in resolved:
        equity *= (1 + t.pnl_pct / 100)
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
    max_hours: int = 168,
    approximate: bool = False,
    seed: int | None = None,
    session: BacktestExecutionSession | None = None,
    prepared_context: PreparedMarketContext | None = None,
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
            max_hours=max_hours,
            approximate=approximate,
            rng=rng,
            session=session,
        )
        trades.append(result)

    trades.sort(key=lambda t: t.entry_time)
    stats = _compute_stats(trades)
    return BacktestResult(trades=trades, **stats)


def backtest_portfolio(
    signals: list[Signal],
    *,
    max_concurrent: int,
    max_per_symbol: int = 1,
    priority_key: Callable[[Signal], Any] | None = None,
    client: BinanceClient | None = None,
    max_hours: int = 168,
    approximate: bool = False,
    seed: int | None = None,
    session: BacktestExecutionSession | None = None,
    prepared_context: PreparedMarketContext | None = None,
    legacy_scheduler: bool = False,
) -> PortfolioResult:
    """Backtest signals with portfolio-level position limits.

    The default staged path schedules the portfolio with approximate
    candle-based trades first, then resolves only accepted trades exactly.
    Set ``legacy_scheduler=True`` to keep the previous resolve-while-scheduling
    behavior as a compatibility fallback.

    Args:
        signals: All candidate signals (any order).
        max_concurrent: Max total open positions at any time.
        max_per_symbol: Max positions per symbol at any time.
        priority_key: Callable(Signal) -> sortable value (lower = higher priority).
            When None, simultaneous signals are shuffled randomly.
        client: BinanceClient instance (created if not provided).
        max_hours: Max trade duration forwarded to backtest_signal().
        seed: RNG seed for reproducible random ordering.
    """
    if legacy_scheduler:
        return _backtest_portfolio_legacy(
            signals,
            max_concurrent=max_concurrent,
            max_per_symbol=max_per_symbol,
            priority_key=priority_key,
            client=client,
            max_hours=max_hours,
            approximate=approximate,
            seed=seed,
            session=session,
            prepared_context=prepared_context,
        )

    session = _ensure_session(
        signal=signals[0] if signals else None,
        client=client,
        session=session,
        prepared_context=prepared_context,
        use_chunk_cache=True,
    )

    sorted_signals = sorted(signals, key=lambda s: s.signal_date)

    open_positions: list[TradeResult] = []
    accepted_signals: list[Signal] = []
    scheduled_trades: list[TradeResult] = []
    skipped: list[SkippedSignal] = []
    utilization_samples: list[float] = []
    peak_util = 0
    rng = _random_module.Random(seed)

    for _date, batch_iter in groupby(sorted_signals, key=lambda s: s.signal_date):
        batch = list(batch_iter)
        signal_date = batch[0].signal_date

        # Expire closed positions
        open_positions = [t for t in open_positions if t.exit_time > signal_date]

        # Record utilization
        current_open = len(open_positions)
        utilization_samples.append(current_open / max_concurrent)
        peak_util = max(peak_util, current_open)

        # Rank batch
        if priority_key is not None:
            batch.sort(key=priority_key)
        else:
            rng.shuffle(batch)

        # Available capacity
        global_available = max_concurrent - current_open
        symbol_counts = Counter(t.signal.ticker for t in open_positions)

        for signal in batch:
            # Check global capacity
            if global_available <= 0:
                skipped.append(SkippedSignal(
                    signal=signal,
                    reason=SkipReason.MAX_CONCURRENT,
                    open_positions=len(open_positions),
                    symbol_positions=symbol_counts.get(signal.ticker, 0),
                ))
                continue

            # Check per-symbol capacity
            if symbol_counts.get(signal.ticker, 0) >= max_per_symbol:
                skipped.append(SkippedSignal(
                    signal=signal,
                    reason=SkipReason.MAX_PER_SYMBOL,
                    open_positions=len(open_positions),
                    symbol_positions=symbol_counts[signal.ticker],
                ))
                continue

            # Accept and schedule with the candle-only path first.
            trade = backtest_signal(
                signal,
                max_hours=max_hours,
                approximate=True,
                rng=rng,
                session=session,
            )
            accepted_signals.append(signal)
            scheduled_trades.append(trade)

            if trade.exit_reason is not ExitReason.UNFILLED:
                open_positions.append(trade)
                global_available -= 1
                symbol_counts[signal.ticker] = symbol_counts.get(signal.ticker, 0) + 1
                peak_util = max(peak_util, len(open_positions))

    if approximate:
        accepted_trades = scheduled_trades
    else:
        accepted_trades = [
            backtest_signal(
                signal,
                max_hours=max_hours,
                approximate=False,
                rng=rng,
                session=session,
            )
            for signal in accepted_signals
        ]

    accepted_trades.sort(key=lambda t: t.entry_time)
    stats = _compute_stats(accepted_trades)

    skipped_concurrent = sum(1 for s in skipped if s.reason is SkipReason.MAX_CONCURRENT)
    skipped_symbol = sum(1 for s in skipped if s.reason is SkipReason.MAX_PER_SYMBOL)
    avg_util = sum(utilization_samples) / len(utilization_samples) if utilization_samples else 0.0

    return PortfolioResult(
        trades=accepted_trades,
        **stats,
        skipped=skipped,
        total_signals=len(signals),
        accepted_signals=len(accepted_trades),
        skipped_by_concurrent=skipped_concurrent,
        skipped_by_symbol=skipped_symbol,
        max_concurrent=max_concurrent,
        max_per_symbol=max_per_symbol,
        peak_utilization=peak_util,
        avg_utilization=avg_util,
    )


def _backtest_portfolio_legacy(
    signals: list[Signal],
    *,
    max_concurrent: int,
    max_per_symbol: int,
    priority_key: Callable[[Signal], Any] | None,
    client: BinanceClient | None,
    max_hours: int,
    approximate: bool,
    seed: int | None,
    session: BacktestExecutionSession | None,
    prepared_context: PreparedMarketContext | None,
) -> PortfolioResult:
    session = _ensure_session(
        signal=signals[0] if signals else None,
        client=client,
        session=session,
        prepared_context=prepared_context,
        use_chunk_cache=True,
    )

    sorted_signals = sorted(signals, key=lambda s: s.signal_date)

    open_positions: list[TradeResult] = []
    accepted_trades: list[TradeResult] = []
    skipped: list[SkippedSignal] = []
    utilization_samples: list[float] = []
    peak_util = 0
    rng = _random_module.Random(seed)

    for _date, batch_iter in groupby(sorted_signals, key=lambda s: s.signal_date):
        batch = list(batch_iter)
        signal_date = batch[0].signal_date

        open_positions = [t for t in open_positions if t.exit_time > signal_date]

        current_open = len(open_positions)
        utilization_samples.append(current_open / max_concurrent)
        peak_util = max(peak_util, current_open)

        if priority_key is not None:
            batch.sort(key=priority_key)
        else:
            rng.shuffle(batch)

        global_available = max_concurrent - current_open
        symbol_counts = Counter(t.signal.ticker for t in open_positions)

        for signal in batch:
            if global_available <= 0:
                skipped.append(
                    SkippedSignal(
                        signal=signal,
                        reason=SkipReason.MAX_CONCURRENT,
                        open_positions=len(open_positions),
                        symbol_positions=symbol_counts.get(signal.ticker, 0),
                    )
                )
                continue

            if symbol_counts.get(signal.ticker, 0) >= max_per_symbol:
                skipped.append(
                    SkippedSignal(
                        signal=signal,
                        reason=SkipReason.MAX_PER_SYMBOL,
                        open_positions=len(open_positions),
                        symbol_positions=symbol_counts[signal.ticker],
                    )
                )
                continue

            trade = backtest_signal(
                signal,
                max_hours=max_hours,
                approximate=approximate,
                rng=rng,
                session=session,
            )
            accepted_trades.append(trade)

            if trade.exit_reason is not ExitReason.UNFILLED:
                open_positions.append(trade)
                global_available -= 1
                symbol_counts[signal.ticker] = symbol_counts.get(signal.ticker, 0) + 1
                peak_util = max(peak_util, len(open_positions))

    accepted_trades.sort(key=lambda t: t.entry_time)
    stats = _compute_stats(accepted_trades)

    skipped_concurrent = sum(1 for s in skipped if s.reason is SkipReason.MAX_CONCURRENT)
    skipped_symbol = sum(1 for s in skipped if s.reason is SkipReason.MAX_PER_SYMBOL)
    avg_util = sum(utilization_samples) / len(utilization_samples) if utilization_samples else 0.0

    return PortfolioResult(
        trades=accepted_trades,
        **stats,
        skipped=skipped,
        total_signals=len(signals),
        accepted_signals=len(accepted_trades),
        skipped_by_concurrent=skipped_concurrent,
        skipped_by_symbol=skipped_symbol,
        max_concurrent=max_concurrent,
        max_per_symbol=max_per_symbol,
        peak_utilization=peak_util,
        avg_utilization=avg_util,
    )


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
    )
