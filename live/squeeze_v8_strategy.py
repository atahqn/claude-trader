"""Live signal generator: Squeeze V8.1 Strategy (SHORT + LONG).

V8.1 extends max holding time from 24h to 72h. All signal logic unchanged
from V8. The change is purely an execution parameter: timeout trades at 24h
were 75% WR and +39.81% PNL across 39 evaluation windows. Extending to 72h
lets those trades resolve to their TP naturally.

V8.1 vs V8 on evaluation windows (11w):
  V8.1 (72h): +152.58% PNL, 81.8% weekly WR, PF 2.39
  V8   (24h): +138.74% PNL, 81.8% weekly WR, PF 2.34

V8 base changes from V7:
  SHORT: TP 2.0→3.0, SL 1.0→1.5, RSI floor 30→25, ATR gate 1.3→1.5
  LONG:  unchanged (TP 4.0, SL 2.0, regime >= 6%)

Architecture:
  Two sub-signals from the SAME structural event (BB/KC squeeze release):

  1. SQUEEZE SHORT:
    - 7+ bar compression, release with negative momentum
    - RSI >= 25 (not deeply oversold), ATR ratio <= 1.5
    - TP: 3.0%, SL: 1.5%, Cooldown: 12h

  2. SQUEEZE LONG:
    - 7+ bar compression, release with POSITIVE momentum
    - ret_72h >= 6% (STRONG BULL regime only)
    - RSI <= 70 (not overbought), ATR ratio <= 1.5
    - TP: 4.0%, SL: 2.0%, Cooldown: 12h

  Max holding time: 72h (V8.1 change)

Polling:
  - analysis interval stays 1h
  - poll interval defaults to 1h, preserving the old close-only behavior
  - when poll interval is lower (for example 15m or 5m), the strategy evaluates
    Binance's current in-progress 1h candle snapshot at those poll boundaries
  - uses incremental _SqueezeV8PreviewState per symbol for O(1) indicator updates

Look-ahead bias prevention:
  - close-only mode uses the last fully closed 1h candle
  - preview mode uses the currently visible 1h candle snapshot from Binance
  - all indicators remain backward-looking: rolling(), ewm(), pct_change()
  - entry executes via market order after signal
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from marketdata import MarketDataRequest

from backtester.indicators import (
    compute_indicator_frame,
    compute_rsi_from_ewm_means,
    linear_regression_slope,
    required_warmup,
    true_range_value,
)
from backtester.models import Candle, MarketType, Signal
from backtester.pipeline import PreparedMarketContext
from backtester.preview import (
    SourcePeriodGate,
    floor_boundary,
    interval_to_seconds,
    iter_preview_snapshots,
)
from backtester.squeeze_signals import emit_squeeze_entry_signals

from .auth_client import LiveMarketClient
from .signal_generator import SignalGenerator

SQUEEZE_V8_SYMBOLS = [
    "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
    "ENA/USDT", "INJ/USDT", "NEAR/USDT", "ALGO/USDT",
    "RENDER/USDT", "WIF/USDT", "ADA/USDT", "APT/USDT",
]
SYMBOLS = SQUEEZE_V8_SYMBOLS

SQUEEZE_V8_FEATURE_COLUMNS = (
    "rsi_14",
    "atr_14",
    "atr_72_avg",
    "atr_ratio",
    "ret_72h",
    "squeeze_on",
    "squeeze_count",
    "mom_slope",
)
_LOOKBACK_BARS = 100
_SUPPORTED_POLL_INTERVALS = {"1h", "30m", "15m", "5m", "1m"}
_RSI_ALPHA = 1 / 14
_EMA20_ALPHA = 2 / 21
_WARMUP_BARS = required_warmup(SQUEEZE_V8_FEATURE_COLUMNS)


@dataclass(slots=True, frozen=True)
class SqueezeV8Config:
    analysis_interval: str = "1h"
    poll_interval: str | None = None
    leverage: float = 1.0
    enable_short: bool = True
    enable_long: bool = True
    short_tp: float = 3.0
    short_sl: float = 1.5
    short_cooldown_h: float = 12.0
    short_rsi_floor: float = 25.0
    long_tp: float = 4.0
    long_sl: float = 2.0
    long_cooldown_h: float = 12.0
    long_rsi_cap: float = 70.0
    long_regime_min: float = 6.0
    min_squeeze_bars: int = 7
    atr_ratio_max: float = 1.5
    taker_fee_rate: float = 0.0005
    market_type: MarketType = MarketType.FUTURES
    max_holding_hours: int = 72
    warmup_bars: int = _WARMUP_BARS

    def __post_init__(self) -> None:
        if self.analysis_interval != "1h":
            raise ValueError("SqueezeV8 currently supports only 1h analysis_interval")
        effective_poll_interval = self.effective_poll_interval
        if effective_poll_interval not in _SUPPORTED_POLL_INTERVALS:
            raise ValueError(
                "SqueezeV8 poll_interval must be one of 1h, 30m, 15m, 5m, 1m"
            )
        if interval_to_seconds(effective_poll_interval) > interval_to_seconds(self.analysis_interval):
            raise ValueError("poll_interval must be less than or equal to analysis_interval")
        if interval_to_seconds(self.analysis_interval) % interval_to_seconds(effective_poll_interval) != 0:
            raise ValueError("poll_interval must divide analysis_interval exactly")

    @property
    def effective_poll_interval(self) -> str:
        return self.poll_interval or self.analysis_interval


def market_data_request_for_squeeze_v8(
    config: SqueezeV8Config | None = None,
) -> MarketDataRequest:
    active_config = config or SqueezeV8Config()
    return MarketDataRequest.ohlcv_only(
        interval=active_config.analysis_interval,
        poll_interval=(
            None
            if active_config.effective_poll_interval == active_config.analysis_interval
            else active_config.effective_poll_interval
        ),
    )


@dataclass(slots=True, frozen=True)
class _SqueezeV8FeatureRow:
    open_time: datetime
    close_time: datetime
    rsi_14: float
    atr_14: float
    atr_72_avg: float
    atr_ratio: float
    ret_72h: float
    squeeze_on: bool
    squeeze_count: int
    mom_slope: float


@dataclass(slots=True, frozen=True)
class _SqueezeV8PreviewStep:
    row: _SqueezeV8FeatureRow
    close: float
    gain_num: float
    gain_den: float
    gain_count: int
    loss_num: float
    loss_den: float
    loss_count: int
    ema_num: float
    ema_den: float
    ema_count: int
    tr_value: float
    atr_14: float


@dataclass(slots=True)
class _SqueezeV8PreviewState:
    candle_count: int = 0
    prev_close: float | None = None
    gain_num: float = 0.0
    gain_den: float = 0.0
    gain_count: int = 0
    loss_num: float = 0.0
    loss_den: float = 0.0
    loss_count: int = 0
    ema_num: float = 0.0
    ema_den: float = 0.0
    ema_count: int = 0
    tr_tail: deque[float] = None  # type: ignore[assignment]
    atr_tail: deque[float] = None  # type: ignore[assignment]
    close_tail_19: deque[float] = None  # type: ignore[assignment]
    close_tail_72: deque[float] = None  # type: ignore[assignment]
    last_row: _SqueezeV8FeatureRow | None = None

    def __post_init__(self) -> None:
        self.tr_tail = deque(maxlen=13)
        self.atr_tail = deque(maxlen=71)
        self.close_tail_19 = deque(maxlen=19)
        self.close_tail_72 = deque(maxlen=72)

    def preview(self, candle: Candle) -> _SqueezeV8PreviewStep:
        close = candle.close
        prev_close = self.prev_close

        gain_mean = np.nan
        gain_num = self.gain_num
        gain_den = self.gain_den
        gain_count = self.gain_count

        loss_mean = np.nan
        loss_num = self.loss_num
        loss_den = self.loss_den
        loss_count = self.loss_count

        if prev_close is not None:
            delta = close - prev_close
            gain_mean, gain_num, gain_den, gain_count = _ewm_next(
                self.gain_num,
                self.gain_den,
                self.gain_count,
                max(delta, 0.0),
                alpha=_RSI_ALPHA,
                min_periods=14,
            )
            loss_mean, loss_num, loss_den, loss_count = _ewm_next(
                self.loss_num,
                self.loss_den,
                self.loss_count,
                max(-delta, 0.0),
                alpha=_RSI_ALPHA,
                min_periods=14,
            )

        rsi = compute_rsi_from_ewm_means(gain_mean, loss_mean)

        tr_value = true_range_value(candle.high, candle.low, prev_close)
        atr_14 = np.nan
        if len(self.tr_tail) == 13:
            atr_14 = (sum(self.tr_tail) + tr_value) / 14.0

        atr_72_avg = np.nan
        if not np.isnan(atr_14) and len(self.atr_tail) == 71:
            atr_72_avg = (sum(self.atr_tail) + atr_14) / 72.0

        atr_ratio = np.nan
        if not np.isnan(atr_14) and not np.isnan(atr_72_avg) and atr_72_avg != 0.0:
            atr_ratio = atr_14 / atr_72_avg

        ret_72h = np.nan
        if len(self.close_tail_72) == 72 and self.close_tail_72[0] != 0.0:
            ret_72h = (close / self.close_tail_72[0] - 1.0) * 100.0

        bb_upper = np.nan
        bb_lower = np.nan
        mom_slope = np.nan
        if len(self.close_tail_19) == 19:
            closes_20 = list(self.close_tail_19) + [close]
            bb_mean = float(np.mean(closes_20))
            bb_std = float(np.std(closes_20, ddof=1))
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            mom_slope = linear_regression_slope(closes_20)

        ema_20, ema_num, ema_den, ema_count = _ewm_next(
            self.ema_num,
            self.ema_den,
            self.ema_count,
            close,
            alpha=_EMA20_ALPHA,
            min_periods=1,
        )
        kc_upper = ema_20 + 1.5 * atr_14 if not np.isnan(atr_14) else np.nan
        kc_lower = ema_20 - 1.5 * atr_14 if not np.isnan(atr_14) else np.nan

        squeeze_on = bool(
            not np.isnan(bb_lower)
            and not np.isnan(kc_lower)
            and bb_lower > kc_lower
            and bb_upper < kc_upper
        )
        previous_squeeze_count = self.last_row.squeeze_count if self.last_row is not None else 0
        squeeze_count = previous_squeeze_count + 1 if squeeze_on else 0

        row = _SqueezeV8FeatureRow(
            open_time=candle.open_time,
            close_time=candle.close_time,
            rsi_14=rsi,
            atr_14=atr_14,
            atr_72_avg=atr_72_avg,
            atr_ratio=atr_ratio,
            ret_72h=ret_72h,
            squeeze_on=squeeze_on,
            squeeze_count=squeeze_count,
            mom_slope=mom_slope,
        )
        return _SqueezeV8PreviewStep(
            row=row,
            close=close,
            gain_num=gain_num,
            gain_den=gain_den,
            gain_count=gain_count,
            loss_num=loss_num,
            loss_den=loss_den,
            loss_count=loss_count,
            ema_num=ema_num,
            ema_den=ema_den,
            ema_count=ema_count,
            tr_value=tr_value,
            atr_14=atr_14,
        )

    def commit(
        self,
        candle: Candle,
        step: _SqueezeV8PreviewStep | None = None,
    ) -> _SqueezeV8FeatureRow:
        if step is None:
            step = self.preview(candle)
        self.candle_count += 1
        self.prev_close = step.close
        self.gain_num = step.gain_num
        self.gain_den = step.gain_den
        self.gain_count = step.gain_count
        self.loss_num = step.loss_num
        self.loss_den = step.loss_den
        self.loss_count = step.loss_count
        self.ema_num = step.ema_num
        self.ema_den = step.ema_den
        self.ema_count = step.ema_count
        self.tr_tail.append(step.tr_value)
        if not np.isnan(step.atr_14):
            self.atr_tail.append(step.atr_14)
        self.close_tail_19.append(step.close)
        self.close_tail_72.append(step.close)
        self.last_row = step.row
        return step.row


def _ewm_next(
    prev_num: float,
    prev_den: float,
    prev_count: int,
    value: float,
    *,
    alpha: float,
    min_periods: int,
) -> tuple[float, float, float, int]:
    beta = 1.0 - alpha
    next_num = value + beta * prev_num
    next_den = 1.0 + beta * prev_den
    next_count = prev_count + 1
    mean = next_num / next_den if next_count >= min_periods else np.nan
    return mean, next_num, next_den, next_count


def build_squeeze_v8_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return compute_indicator_frame(frame, SQUEEZE_V8_FEATURE_COLUMNS)


def build_squeeze_v8_feature_frames(
    prepared_context: PreparedMarketContext,
    *,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    feature_frames: dict[str, pd.DataFrame] = {}
    selected_symbols = symbols or prepared_context.symbols
    for symbol in selected_symbols:
        feature_frames[symbol] = build_squeeze_v8_feature_frame(
            prepared_context.for_symbol(symbol).frame
        )
    return feature_frames


def _safe_value(row: pd.Series, column: str, default: float) -> float:
    value = row.get(column, default)
    return default if pd.isna(value) else float(value)


def _safe_feature_value(value: float, default: float) -> float:
    return default if np.isnan(value) else float(value)


def _generate_symbol_signals(
    frame: pd.DataFrame,
    symbol: str,
    start: datetime,
    end: datetime,
    config: SqueezeV8Config,
) -> list[Signal]:
    if frame.empty:
        return []

    signals: list[Signal] = []
    last_short: datetime | None = None
    last_long: datetime | None = None

    first_eval_index = max(min(config.warmup_bars - 1, len(frame) - 1), 1)
    for index in range(first_eval_index, len(frame)):
        row = frame.iloc[index]
        prev = frame.iloc[index - 1]
        close_time = row["close_time"]
        if close_time < start or close_time >= end:
            continue

        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            continue

        atr_ratio = _safe_value(row, "atr_ratio", 1.0)
        if atr_ratio > config.atr_ratio_max:
            continue

        if _safe_value(prev, "squeeze_count", 0.0) < config.min_squeeze_bars:
            continue
        if bool(row.get("squeeze_on", False)):
            continue

        rsi = _safe_value(row, "rsi_14", 50.0)
        ret_72h = _safe_value(row, "ret_72h", 0.0)
        mom = _safe_value(row, "mom_slope", 0.0)
        sq_count = int(_safe_value(prev, "squeeze_count", 0.0))
        emitted, last_short, last_long = emit_squeeze_entry_signals(
            signal_date=close_time,
            symbol=symbol,
            config=config,
            strategy_name="squeeze_v8.1",
            prev_squeeze_count=sq_count,
            squeeze_on=bool(row.get("squeeze_on", False)),
            mom=mom,
            rsi=rsi,
            atr_ratio=atr_ratio,
            ret_72h=ret_72h,
            last_short=last_short,
            last_long=last_long,
        )
        signals.extend(emitted)

    return signals


def _generate_symbol_preview_signals(
    poll_candles: list[Candle],
    symbol: str,
    start: datetime,
    end: datetime,
    config: SqueezeV8Config,
) -> list[Signal]:
    if not poll_candles:
        return []

    signals: list[Signal] = []
    state = _SqueezeV8PreviewState()
    gate = SourcePeriodGate()
    last_short: datetime | None = None
    last_long: datetime | None = None
    warmup_bars = max(config.warmup_bars, 1)

    for snapshot in iter_preview_snapshots(poll_candles, config.analysis_interval):
        partial_candle = snapshot.candle

        if snapshot.skipped_candle is not None:
            state.commit(snapshot.skipped_candle)

        step: _SqueezeV8PreviewStep | None = None
        last_row = state.last_row
        state_ready = (
            last_row is not None
            and (
                state.candle_count >= warmup_bars
                or (not np.isnan(last_row.mom_slope) and not np.isnan(last_row.atr_ratio))
            )
        )
        if state_ready and start <= snapshot.signal_time < end:
            step = state.preview(partial_candle)
            row = step.row
            prev = state.last_row

            if not np.isnan(row.mom_slope) and not np.isnan(row.atr_ratio):
                atr_ratio = _safe_feature_value(row.atr_ratio, 1.0)
                if atr_ratio <= config.atr_ratio_max:
                    prev_sq_count = float(prev.squeeze_count)
                    if prev_sq_count >= config.min_squeeze_bars and not row.squeeze_on:
                        rsi = _safe_feature_value(row.rsi_14, 50.0)
                        ret_72h = _safe_feature_value(row.ret_72h, 0.0)
                        mom = _safe_feature_value(row.mom_slope, 0.0)
                        sq_count = int(prev_sq_count)
                        emitted, last_short, last_long = emit_squeeze_entry_signals(
                            signal_date=snapshot.signal_time,
                            symbol=symbol,
                            config=config,
                            strategy_name="squeeze_v8.1",
                            prev_squeeze_count=sq_count,
                            squeeze_on=row.squeeze_on,
                            mom=mom,
                            rsi=rsi,
                            atr_ratio=atr_ratio,
                            ret_72h=ret_72h,
                            last_short=last_short,
                            last_long=last_long,
                            source_period_start=snapshot.source_period_start,
                            source_period_gate=gate,
                        )
                        signals.extend(emitted)

        if snapshot.is_final:
            state.commit(partial_candle, step=step)

    return signals


def generate_squeeze_v8_signals(
    prepared_context: PreparedMarketContext,
    *,
    config: SqueezeV8Config | None = None,
    symbols: list[str] | None = None,
    feature_frames: dict[str, pd.DataFrame] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Signal]:
    active_config = config or SqueezeV8Config()
    active_start = start if start is not None else prepared_context.start
    active_end = end if end is not None else prepared_context.end
    if active_config.effective_poll_interval != active_config.analysis_interval:
        if feature_frames is not None:
            raise ValueError("feature_frames are not supported with preview polling")
        if prepared_context.request.effective_poll_ohlcv_interval != active_config.effective_poll_interval:
            raise ValueError(
                "prepared_context poll interval does not match SqueezeV8Config.poll_interval"
            )
        selected_symbols = symbols or prepared_context.symbols
        signals: list[Signal] = []
        for symbol in selected_symbols:
            signals.extend(
                _generate_symbol_preview_signals(
                    prepared_context.slice_poll_candles(
                        symbol,
                        prepared_context.fetch_start,
                        active_end,
                    ),
                    symbol,
                    active_start,
                    active_end,
                    active_config,
                )
            )
        return sorted(signals, key=lambda signal: signal.signal_date)

    frames = feature_frames or build_squeeze_v8_feature_frames(
        prepared_context,
        symbols=symbols,
    )
    selected_symbols = symbols or list(frames)

    signals: list[Signal] = []
    for symbol in selected_symbols:
        frame = frames.get(symbol)
        if frame is None:
            continue
        signals.extend(
            _generate_symbol_signals(
                frame,
                symbol,
                active_start,
                active_end,
                active_config,
            )
        )
    return sorted(signals, key=lambda signal: signal.signal_date)


class SqueezeV8Strategy(SignalGenerator):
    """Squeeze V8.1: SHORT + LONG live signal generator.

    SHORT: squeeze breakout with negative momentum (all regimes)
           TP 3.0/SL 1.5, RSI >= 25, ATR ratio <= 1.5
    LONG:  squeeze breakout with positive momentum (bull regime only, ret_72h >= 6%)
           TP 4.0/SL 2.0, RSI <= 70, ATR ratio <= 1.5
    Max holding time: 72h (V8.1 change from 24h default)
    """

    def __init__(
        self,
        analysis_interval: str = "1h",
        poll_interval: str | None = None,
        leverage: float = 1.0,
        # SHORT params (V8: wider TP/SL, lower RSI floor)
        short_tp: float = 3.0,
        short_sl: float = 1.5,
        short_cooldown_h: float = 12.0,
        short_rsi_floor: float = 25.0,
        # LONG params (unchanged from V7)
        long_tp: float = 4.0,
        long_sl: float = 2.0,
        long_cooldown_h: float = 12.0,
        long_rsi_cap: float = 70.0,
        long_regime_min: float = 6.0,
        # Shared params (V8: relaxed ATR gate)
        min_squeeze_bars: int = 7,
        atr_ratio_max: float = 1.5,
        # V8.1: extended max holding time
        max_holding_hours: int = 72,
    ) -> None:
        if analysis_interval != "1h":
            raise ValueError("SqueezeV8 live strategy currently supports only 1h analysis_interval")
        effective_poll_interval = poll_interval or analysis_interval
        if effective_poll_interval not in _SUPPORTED_POLL_INTERVALS:
            raise ValueError("poll_interval must be one of 1h, 30m, 15m, 5m, 1m")
        if interval_to_seconds(analysis_interval) % interval_to_seconds(effective_poll_interval) != 0:
            raise ValueError("poll_interval must divide analysis_interval exactly")

        self.analysis_interval = analysis_interval
        self.poll_interval = poll_interval
        self.enable_short = True
        self.enable_long = True
        self.leverage = leverage
        self.short_tp = short_tp
        self.short_sl = short_sl
        self.short_cooldown_h = short_cooldown_h
        self.short_rsi_floor = short_rsi_floor
        self.long_tp = long_tp
        self.long_sl = long_sl
        self.long_cooldown_h = long_cooldown_h
        self.long_rsi_cap = long_rsi_cap
        self.long_regime_min = long_regime_min
        self.min_squeeze_bars = min_squeeze_bars
        self.atr_ratio_max = atr_ratio_max
        self.max_holding_hours = max_holding_hours

        self._client: LiveMarketClient | None = None
        self._last_short: dict[str, datetime] = {}
        self._last_long: dict[str, datetime] = {}
        self._source_period_gate = SourcePeriodGate()
        self._states: dict[str, _SqueezeV8PreviewState] = {}
        self._last_committed_hour: dict[str, datetime] = {}

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        self._warm_up_states()
        print(
            f"SqueezeV8.1 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"analysis={self.analysis_interval} | poll={self.effective_poll_interval} | "
            f"SHORT TP/SL={self.short_tp}/{self.short_sl}% "
            f"CD={self.short_cooldown_h}h RSI>={self.short_rsi_floor} | "
            f"LONG TP/SL={self.long_tp}/{self.long_sl}% "
            f"CD={self.long_cooldown_h}h RSI<={self.long_rsi_cap} "
            f"regime>={self.long_regime_min}% | "
            f"min_sq={self.min_squeeze_bars} ATR<={self.atr_ratio_max} | "
            f"max_hold={self.max_holding_hours}h",
            file=sys.stderr,
        )

    def _warm_up_states(self) -> None:
        assert self._client is not None
        now = datetime.now(UTC)
        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        for symbol in SYMBOLS:
            try:
                candles = self._client.fetch_klines(
                    symbol=symbol.replace("/", ""),
                    interval=self.analysis_interval,
                    start=start,
                    end=now,
                )
                state = _SqueezeV8PreviewState()
                for candle in candles:
                    if candle.close_time <= now:
                        state.commit(candle)
                        self._last_committed_hour[symbol] = candle.open_time
                self._states[symbol] = state
                print(
                    f"  {symbol}: warmed up {state.candle_count} bars",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"  {symbol}: warmup failed: {exc}", file=sys.stderr)
                self._states[symbol] = _SqueezeV8PreviewState()

    def poll(self) -> list[Signal] | None:
        assert self._client is not None
        now = self.current_time()
        signals: list[Signal] = []

        for symbol in SYMBOLS:
            try:
                sigs = self._check_symbol(symbol, now)
                signals.extend(sigs)
            except Exception as exc:
                print(f"Error checking {symbol}: {exc}", file=sys.stderr)

        return signals if signals else None

    def _check_symbol(self, symbol: str, now: datetime) -> list[Signal]:
        assert self._client is not None

        state = self._states.get(symbol)
        if state is None:
            return []

        preview_enabled = self.effective_poll_interval != self.analysis_interval

        # Commit any newly closed hourly candles since last commit
        last_committed = self._last_committed_hour.get(symbol)
        current_hour = floor_boundary(now, self.analysis_interval)
        if last_committed is None or last_committed < current_hour:
            new_candles = self._client.fetch_klines(
                symbol=symbol.replace("/", ""),
                interval=self.analysis_interval,
                start=(last_committed or current_hour - timedelta(hours=2)),
                end=now,
            )
            for candle in new_candles:
                candle_hour = candle.open_time
                if candle.close_time <= now and (
                    last_committed is None or candle_hour > last_committed
                ):
                    state.commit(candle)
                    self._last_committed_hour[symbol] = candle_hour

        if state.last_row is None:
            return []
        if (
            state.candle_count < _WARMUP_BARS
            and (np.isnan(state.last_row.mom_slope) or np.isnan(state.last_row.atr_ratio))
        ):
            return []

        # Get the current candle to evaluate
        if preview_enabled:
            # Fetch the current open 1h candle snapshot from Binance
            candles = self._client.fetch_klines(
                symbol=symbol.replace("/", ""),
                interval=self.analysis_interval,
                start=current_hour,
                end=now + timedelta(hours=1),
            )
            if not candles:
                return []
            eval_candle = candles[-1]
            if eval_candle.close_time <= now:
                # Hour already closed, was committed above — nothing to preview
                return []
        else:
            # Non-preview: evaluate the last committed candle
            if state.last_row is None:
                return []
            eval_candle = Candle(
                open_time=state.last_row.open_time,
                close_time=state.last_row.close_time,
                open=0.0,
                high=0.0,
                low=0.0,
                close=state.prev_close or 0.0,
                volume=0.0,
            )
            # For non-preview, use the committed last_row directly
            row = state.last_row
            prev_row = state.last_row  # Need the row before last_row
            # Non-preview path: we need the previous bar's squeeze_count.
            # Since we only have the last committed row, we use it as "current"
            # and need the one before it. Fall back to pandas for non-preview.
            return self._check_symbol_pandas(symbol, now)

        # Preview path: evaluate partial candle with incremental state
        step = state.preview(eval_candle)
        row = step.row
        prev = state.last_row

        if np.isnan(row.mom_slope) or np.isnan(row.atr_ratio):
            return []

        atr_ratio = _safe_feature_value(row.atr_ratio, 1.0)
        if atr_ratio > self.atr_ratio_max:
            return []

        if prev.squeeze_count < self.min_squeeze_bars or row.squeeze_on:
            return []

        rsi = _safe_feature_value(row.rsi_14, 50.0)
        ret_72h = _safe_feature_value(row.ret_72h, 0.0)
        mom = _safe_feature_value(row.mom_slope, 0.0)

        source_hour_start = eval_candle.open_time
        emitted, self._last_short[symbol], self._last_long[symbol] = emit_squeeze_entry_signals(
            signal_date=now,
            symbol=symbol,
            config=self,
            strategy_name="squeeze_v8.1",
            prev_squeeze_count=int(prev.squeeze_count),
            squeeze_on=row.squeeze_on,
            mom=mom,
            rsi=rsi,
            atr_ratio=atr_ratio,
            ret_72h=ret_72h,
            last_short=self._last_short.get(symbol),
            last_long=self._last_long.get(symbol),
            source_period_start=source_hour_start,
            source_period_gate=self._source_period_gate,
            short_gate_key=symbol,
            long_gate_key=symbol,
        )
        return emitted

    def _check_symbol_pandas(self, symbol: str, now: datetime) -> list[Signal]:
        """Fallback for non-preview (hourly) polling: full pandas recompute.

        Only used when poll_interval == analysis_interval (default 1h mode).
        """
        assert self._client is not None
        import pandas as pd

        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval=self.analysis_interval,
            start=start,
            end=now,
        )

        if len(candles) < _WARMUP_BARS:
            return []

        rows = []
        for c in candles:
            rows.append({
                "open_time": c.open_time,
                "close_time": c.close_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return []
        df = df.sort_values("open_time").reset_index(drop=True)
        df = build_squeeze_v8_feature_frame(df)

        last_idx = len(df) - 1
        if df.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx + 1 < _WARMUP_BARS:
            return []

        row = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]

        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            return []

        rsi = row.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50
        atr_ratio = row.get("atr_ratio", 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0
        ret_72h = row.get("ret_72h", 0)
        if pd.isna(ret_72h):
            ret_72h = 0

        if atr_ratio > self.atr_ratio_max:
            return []
        if prev["squeeze_count"] < self.min_squeeze_bars or row["squeeze_on"]:
            return []

        mom = row["mom_slope"]
        emitted, self._last_short[symbol], self._last_long[symbol] = emit_squeeze_entry_signals(
            signal_date=now,
            symbol=symbol,
            config=self,
            strategy_name="squeeze_v8.1",
            prev_squeeze_count=int(prev["squeeze_count"]),
            squeeze_on=bool(row["squeeze_on"]),
            mom=float(mom),
            rsi=float(rsi),
            atr_ratio=float(atr_ratio),
            ret_72h=float(ret_72h),
            last_short=self._last_short.get(symbol),
            last_long=self._last_long.get(symbol),
        )
        return emitted

    # ------------------------------------------------------------------
    # Backtest interface
    # ------------------------------------------------------------------

    def _as_backtest_config(self) -> SqueezeV8Config:
        return SqueezeV8Config(
            analysis_interval=self.analysis_interval,
            poll_interval=self.poll_interval,
            leverage=self.leverage,
            enable_short=self.enable_short,
            enable_long=self.enable_long,
            short_tp=self.short_tp,
            short_sl=self.short_sl,
            short_cooldown_h=self.short_cooldown_h,
            short_rsi_floor=self.short_rsi_floor,
            long_tp=self.long_tp,
            long_sl=self.long_sl,
            long_cooldown_h=self.long_cooldown_h,
            long_rsi_cap=self.long_rsi_cap,
            long_regime_min=self.long_regime_min,
            min_squeeze_bars=self.min_squeeze_bars,
            atr_ratio_max=self.atr_ratio_max,
            max_holding_hours=self.max_holding_hours,
        )

    def generate_backtest_signals(self, prepared_context, symbols, start, end):
        config = self._as_backtest_config()
        if config.effective_poll_interval == config.analysis_interval:
            feats = build_squeeze_v8_feature_frames(
                prepared_context, symbols=symbols,
            )
            return generate_squeeze_v8_signals(
                prepared_context, config=config, symbols=symbols,
                feature_frames=feats, start=start, end=end,
            )
        return generate_squeeze_v8_signals(
            prepared_context, config=config, symbols=symbols,
            start=start, end=end,
        )
