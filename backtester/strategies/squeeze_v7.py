from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..indicators import (
    compute_indicator_frame,
    compute_rsi_from_ewm_means,
    linear_regression_slope,
    required_warmup,
    true_range_value,
)
from ..models import Candle
from ..preview import SourcePeriodGate, interval_to_seconds, iter_preview_snapshots
from ..squeeze_signals import emit_squeeze_entry_signals
from marketdata import MarketDataRequest

from ..models import MarketType, Signal
from ..pipeline import PreparedMarketContext

SQUEEZE_V7_SYMBOLS = [
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "ENA/USDT",
    "INJ/USDT",
    "NEAR/USDT",
    "ALGO/USDT",
    "RENDER/USDT",
    "WIF/USDT",
    "ADA/USDT",
    "APT/USDT",
]

_SUPPORTED_SQUEEZE_INTERVALS = {"1h", "30m", "15m", "5m", "1m"}
_RSI_ALPHA = 1 / 14
_EMA20_ALPHA = 2 / 21
SQUEEZE_V7_FEATURE_COLUMNS = (
    "rsi_14",
    "atr_14",
    "atr_72_avg",
    "atr_ratio",
    "ret_72h",
    "squeeze_on",
    "squeeze_count",
    "mom_slope",
)
_SQUEEZE_V7_WARMUP_BARS = required_warmup(SQUEEZE_V7_FEATURE_COLUMNS)


@dataclass(slots=True, frozen=True)
class SqueezeV7Config:
    analysis_interval: str = "1h"
    poll_interval: str | None = None
    leverage: float = 1.0
    enable_short: bool = True
    enable_long: bool = True
    short_tp: float = 2.0
    short_sl: float = 1.0
    short_cooldown_h: float = 12.0
    short_rsi_floor: float = 30.0
    long_tp: float = 4.0
    long_sl: float = 2.0
    long_cooldown_h: float = 12.0
    long_rsi_cap: float = 70.0
    long_regime_min: float = 6.0
    min_squeeze_bars: int = 7
    atr_ratio_max: float = 1.3
    taker_fee_rate: float = 0.0005
    market_type: MarketType = MarketType.FUTURES
    warmup_bars: int = _SQUEEZE_V7_WARMUP_BARS

    def __post_init__(self) -> None:
        if self.analysis_interval != "1h":
            raise ValueError("SqueezeV7 currently supports only 1h analysis_interval")
        effective_poll_interval = self.effective_poll_interval
        if effective_poll_interval not in _SUPPORTED_SQUEEZE_INTERVALS:
            raise ValueError(
                "SqueezeV7 poll_interval must be one of 1h, 30m, 15m, 5m, 1m"
            )
        if interval_to_seconds(effective_poll_interval) > interval_to_seconds(self.analysis_interval):
            raise ValueError("poll_interval must be less than or equal to analysis_interval")
        if interval_to_seconds(self.analysis_interval) % interval_to_seconds(effective_poll_interval) != 0:
            raise ValueError("poll_interval must divide analysis_interval exactly")

    @property
    def effective_poll_interval(self) -> str:
        return self.poll_interval or self.analysis_interval


def market_data_request_for_squeeze_v7(
    config: SqueezeV7Config | None = None,
) -> MarketDataRequest:
    active_config = config or SqueezeV7Config()
    return MarketDataRequest.ohlcv_only(
        interval=active_config.analysis_interval,
        poll_interval=(
            None
            if active_config.effective_poll_interval == active_config.analysis_interval
            else active_config.effective_poll_interval
        ),
    )

@dataclass(slots=True, frozen=True)
class _SqueezeV7FeatureRow:
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
class _SqueezeV7PreviewStep:
    row: _SqueezeV7FeatureRow
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
class _SqueezeV7PreviewState:
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
    last_row: _SqueezeV7FeatureRow | None = None

    def __post_init__(self) -> None:
        self.tr_tail = deque(maxlen=13)
        self.atr_tail = deque(maxlen=71)
        self.close_tail_19 = deque(maxlen=19)
        self.close_tail_72 = deque(maxlen=72)

    def preview(self, candle: Candle) -> _SqueezeV7PreviewStep:
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

        _ema20, ema_num, ema_den, ema_count = _ewm_next(
            self.ema_num,
            self.ema_den,
            self.ema_count,
            close,
            alpha=_EMA20_ALPHA,
            min_periods=1,
        )
        kc_upper = _ema20 + 1.5 * atr_14 if not np.isnan(atr_14) else np.nan
        kc_lower = _ema20 - 1.5 * atr_14 if not np.isnan(atr_14) else np.nan

        squeeze_on = bool(
            not np.isnan(bb_lower)
            and not np.isnan(kc_lower)
            and bb_lower > kc_lower
            and bb_upper < kc_upper
        )
        previous_squeeze_count = self.last_row.squeeze_count if self.last_row is not None else 0
        squeeze_count = previous_squeeze_count + 1 if squeeze_on else 0

        row = _SqueezeV7FeatureRow(
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
        return _SqueezeV7PreviewStep(
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
        step: _SqueezeV7PreviewStep | None = None,
    ) -> _SqueezeV7FeatureRow:
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


def build_squeeze_v7_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return compute_indicator_frame(frame, SQUEEZE_V7_FEATURE_COLUMNS)


def build_squeeze_v7_feature_frames(
    prepared_context: PreparedMarketContext,
    *,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    feature_frames: dict[str, pd.DataFrame] = {}
    selected_symbols = symbols or prepared_context.symbols
    for symbol in selected_symbols:
        feature_frames[symbol] = build_squeeze_v7_feature_frame(
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
    config: SqueezeV7Config,
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
            strategy_name="squeeze_v7",
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
    config: SqueezeV7Config,
) -> list[Signal]:
    if not poll_candles:
        return []

    signals: list[Signal] = []
    state = _SqueezeV7PreviewState()
    gate = SourcePeriodGate()
    last_short: datetime | None = None
    last_long: datetime | None = None
    warmup_bars = max(config.warmup_bars, 1)

    for snapshot in iter_preview_snapshots(poll_candles, config.analysis_interval):
        partial_candle = snapshot.candle

        if snapshot.skipped_candle is not None:
            state.commit(snapshot.skipped_candle)

        step: _SqueezeV7PreviewStep | None = None
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
                            strategy_name="squeeze_v7",
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


def generate_squeeze_v7_signals(
    prepared_context: PreparedMarketContext,
    *,
    config: SqueezeV7Config | None = None,
    symbols: list[str] | None = None,
    feature_frames: dict[str, pd.DataFrame] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Signal]:
    active_config = config or SqueezeV7Config()
    active_start = start if start is not None else prepared_context.start
    active_end = end if end is not None else prepared_context.end
    if active_config.effective_poll_interval != active_config.analysis_interval:
        if feature_frames is not None:
            raise ValueError("feature_frames are not supported with preview polling")
        if prepared_context.request.effective_poll_ohlcv_interval != active_config.effective_poll_interval:
            raise ValueError(
                "prepared_context poll interval does not match SqueezeV7Config.poll_interval"
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

    frames = feature_frames or build_squeeze_v7_feature_frames(
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
