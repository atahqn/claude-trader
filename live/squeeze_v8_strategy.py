"""Live signal generator: Squeeze V8.2 Strategy (SHORT + LONG).

V8.2 keeps the V8.1 signal set unchanged and adds dynamic position sizing.
Entries, exits, TP/SL, cooldowns, and filters are the same as V8.1. The only
change is capital allocation per trade via `Signal.size_multiplier`.

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
  Position sizing: ridge_v1 dynamic sizing (V8.2 change)

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import math
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

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
from backtester.models import Candle, MarketType, PositionType, Signal
from backtester.pipeline import PreparedMarketContext
from backtester.preview import (
    SourcePeriodGate,
    floor_boundary,
    interval_to_seconds,
    iter_preview_snapshots,
)

from .auth_client import LiveMarketClient
from .signal_generator import FatalSignalError, SignalGenerator

SYMBOLS = [
    "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
    "ENA/USDT", "INJ/USDT", "NEAR/USDT", "ALGO/USDT",
    "RENDER/USDT", "WIF/USDT", "ADA/USDT", "APT/USDT",
]

SQUEEZE_V8_FEATURE_COLUMNS = (
    "rsi_14",
    "atr_ratio",
    "ret_72h",
    "squeeze_on",
    "squeeze_count",
    "mom_slope",
)
_LOOKBACK_BARS = 100
_SUPPORTED_POLL_INTERVALS = {"1h", "30m", "15m", "5m", "1m"}
_SUPPORTED_SIZING_MODES = {"baseline", "ridge_v1"}
_RSI_ALPHA = 1 / 14
_EMA20_ALPHA = 2 / 21
_WARMUP_BARS = required_warmup(SQUEEZE_V8_FEATURE_COLUMNS)

_RIDGE_V1_SHORT_BETA = (
    -0.0797095672793,
    -0.00113339169499,
    -0.00545966581917,
    -0.0481497500225,
    -0.0447666408166,
    0.134452440809,
    -0.0173908941701,
    -0.0956252340052,
    0.02824767755,
    0.0,
    0.0,
    0.473511481157,
    0.0762125236835,
)
_RIDGE_V1_LONG_BETA = (
    -0.162865266543,
    0.0359613159204,
    0.0474623378793,
    -0.0769663844052,
    -0.0156901218046,
    -0.095572751899,
    0.0369201524174,
    0.795102644056,
    -0.153238461595,
    -0.15074135538,
    0.0642680237811,
    0.810609883215,
    0.014871683311,
)
_RIDGE_V1_SHORT_MEAN = 0.2885689587426325
_RIDGE_V1_SHORT_STD = 0.32868843251811886
_RIDGE_V1_LONG_MEAN = 0.8006792452830193
_RIDGE_V1_LONG_STD = 0.5909870620140927
_RIDGE_V1_SHORT_ALPHA = 0.34
_RIDGE_V1_LONG_ALPHA = 0.10
_RIDGE_V1_CLIP_LO = 0.45
_RIDGE_V1_CLIP_HI = 1.7
_RIDGE_V1_SCALE_2 = 0.9964448114052744


def _validate_sizing_mode(mode: str) -> str:
    if mode not in _SUPPORTED_SIZING_MODES:
        supported = ", ".join(sorted(_SUPPORTED_SIZING_MODES))
        raise ValueError(f"sizing_mode must be one of: {supported}")
    return mode


def _signal_metadata_float(signal: Signal, key: str, default: float) -> float:
    value = signal.metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ridge_v1_feature_vector(
    signal: Signal,
    *,
    cluster_side: int,
    cluster_total: int,
) -> tuple[float, ...]:
    sq_count = _signal_metadata_float(signal, "sq_count", 12.0)
    atr_ratio = _signal_metadata_float(signal, "atr_ratio", 0.9)
    mom = abs(_signal_metadata_float(signal, "mom", 0.0))
    rsi = _signal_metadata_float(signal, "rsi", 50.0)
    ret_72h = _signal_metadata_float(signal, "ret_72h", 10.0)

    sq = (sq_count - 12.0) / 4.0
    atr = (atr_ratio - 0.9) / 0.25
    log_mom = math.log1p(max(mom, 0.0) * 1000.0)
    rsi_centered = (rsi - 50.0) / 10.0
    ret = (ret_72h - 10.0) / 5.0
    side_cluster = math.log1p(max(float(cluster_side) - 1.0, 0.0))
    total_cluster = math.log1p(max(float(cluster_total) - 1.0, 0.0))

    return (
        1.0,
        sq,
        sq**2,
        atr,
        atr**2,
        log_mom,
        log_mom**2,
        rsi_centered,
        rsi_centered**2,
        ret,
        ret**2,
        side_cluster,
        total_cluster,
    )


def _ridge_v1_size_multiplier(
    signal: Signal,
    *,
    cluster_side: int,
    cluster_total: int,
) -> float:
    if signal.position_type is PositionType.SHORT:
        beta = _RIDGE_V1_SHORT_BETA
        mean = _RIDGE_V1_SHORT_MEAN
        std = _RIDGE_V1_SHORT_STD
        alpha = _RIDGE_V1_SHORT_ALPHA
    else:
        beta = _RIDGE_V1_LONG_BETA
        mean = _RIDGE_V1_LONG_MEAN
        std = _RIDGE_V1_LONG_STD
        alpha = _RIDGE_V1_LONG_ALPHA

    score = float(
        np.dot(
            np.asarray(beta, dtype=float),
            np.asarray(
                _ridge_v1_feature_vector(
                    signal,
                    cluster_side=cluster_side,
                    cluster_total=cluster_total,
                ),
                dtype=float,
            ),
        )
    )
    z_score = (score - mean) / std
    raw = float(np.clip(1.0 + alpha * z_score, 0.2, 3.0))
    normalized = float(np.clip(raw, _RIDGE_V1_CLIP_LO, _RIDGE_V1_CLIP_HI))
    return float(
        np.clip(
            normalized / _RIDGE_V1_SCALE_2,
            _RIDGE_V1_CLIP_LO,
            _RIDGE_V1_CLIP_HI,
        )
    )


def _apply_dynamic_sizing(
    signals: list[Signal],
    *,
    sizing_mode: str,
) -> list[Signal]:
    if not signals or sizing_mode == "baseline":
        return signals
    _validate_sizing_mode(sizing_mode)

    total_counts: dict[datetime, int] = {}
    side_counts: dict[tuple[datetime, PositionType], int] = {}
    for signal in signals:
        total_counts[signal.signal_date] = total_counts.get(signal.signal_date, 0) + 1
        key = (signal.signal_date, signal.position_type)
        side_counts[key] = side_counts.get(key, 0) + 1

    sized: list[Signal] = []
    for signal in signals:
        cluster_total = total_counts[signal.signal_date]
        cluster_side = side_counts[(signal.signal_date, signal.position_type)]
        if sizing_mode == "ridge_v1":
            size_multiplier = _ridge_v1_size_multiplier(
                signal,
                cluster_side=cluster_side,
                cluster_total=cluster_total,
            )
        else:  # pragma: no cover - guarded above
            raise ValueError(f"unsupported sizing_mode: {sizing_mode}")

        metadata = dict(signal.metadata)
        metadata["size_model"] = sizing_mode
        metadata["cluster_side"] = cluster_side
        metadata["cluster_total"] = cluster_total
        metadata["size_mult"] = round(size_multiplier, 6)
        sized.append(
            replace(
                signal,
                size_multiplier=size_multiplier,
                metadata=metadata,
            )
        )
    return sized


@dataclass(slots=True, frozen=True)
class SqueezeV8Config:
    analysis_interval: str = "1h"
    poll_interval: str | None = None
    sizing_mode: str = "ridge_v1"
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
        _validate_sizing_mode(self.sizing_mode)
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


def _build_squeeze_signal(
    *,
    signal_date: datetime,
    position_type: PositionType,
    symbol: str,
    config: Any,
    metadata: dict[str, object],
    size_multiplier: float = 1.0,
) -> Signal:
    max_holding_hours = getattr(config, "max_holding_hours", 72)
    return Signal(
        signal_date=signal_date,
        position_type=position_type,
        ticker=symbol,
        tp_pct=config.short_tp if position_type is PositionType.SHORT else config.long_tp,
        sl_pct=config.short_sl if position_type is PositionType.SHORT else config.long_sl,
        leverage=config.leverage,
        market_type=getattr(config, "market_type", MarketType.FUTURES),
        taker_fee_rate=getattr(config, "taker_fee_rate", 0.0005),
        max_holding_hours=max_holding_hours,
        size_multiplier=size_multiplier,
        metadata=metadata,
    )


def _emit_squeeze_entry_signals(
    *,
    signal_date: datetime,
    symbol: str,
    config: Any,
    strategy_name: str,
    prev_squeeze_count: int,
    squeeze_on: bool,
    mom: float,
    rsi: float,
    atr_ratio: float,
    ret_72h: float,
    last_short: datetime | None,
    last_long: datetime | None,
    source_period_start: datetime | None = None,
    source_period_gate: SourcePeriodGate | None = None,
    short_gate_key: object = "default",
    long_gate_key: object = "default",
) -> tuple[list[Signal], datetime | None, datetime | None]:
    if atr_ratio > config.atr_ratio_max:
        return [], last_short, last_long
    if prev_squeeze_count < config.min_squeeze_bars or squeeze_on:
        return [], last_short, last_long

    signals: list[Signal] = []
    preview_metadata: dict[str, object] = {}
    if source_period_start is not None:
        preview_metadata = {
            "analysis_interval": getattr(config, "analysis_interval", "1h"),
            "poll_interval": getattr(
                config,
                "effective_poll_interval",
                getattr(config, "analysis_interval", "1h"),
            ),
            "source_hour_start": source_period_start.isoformat(),
        }

    enable_short = getattr(config, "enable_short", True)
    enable_long = getattr(config, "enable_long", True)

    short_gate_ok = (
        source_period_start is None
        or source_period_gate is None
        or source_period_gate.allow(source_period_start, key=short_gate_key)
    )
    if (
        enable_short
        and short_gate_ok
        and mom < 0
        and rsi >= config.short_rsi_floor
        and (
            last_short is None
            or (signal_date - last_short).total_seconds() >= config.short_cooldown_h * 3600
        )
    ):
        last_short = signal_date
        if source_period_start is not None and source_period_gate is not None:
            source_period_gate.mark(source_period_start, key=short_gate_key)
        signals.append(
            _build_squeeze_signal(
                signal_date=signal_date,
                position_type=PositionType.SHORT,
                symbol=symbol,
                config=config,
                metadata={
                    "strategy": f"{strategy_name}_short",
                    **preview_metadata,
                    "mom": round(mom, 6),
                    "rsi": round(rsi, 1),
                    "atr_ratio": round(atr_ratio, 2),
                    "sq_count": int(prev_squeeze_count),
                },
            )
        )

    long_gate_ok = (
        source_period_start is None
        or source_period_gate is None
        or source_period_gate.allow(source_period_start, key=long_gate_key)
    )
    if (
        enable_long
        and long_gate_ok
        and mom > 0
        and rsi <= config.long_rsi_cap
        and ret_72h >= config.long_regime_min
        and (
            last_long is None
            or (signal_date - last_long).total_seconds() >= config.long_cooldown_h * 3600
        )
    ):
        last_long = signal_date
        if source_period_start is not None and source_period_gate is not None:
            source_period_gate.mark(source_period_start, key=long_gate_key)
        signals.append(
            _build_squeeze_signal(
                signal_date=signal_date,
                position_type=PositionType.LONG,
                symbol=symbol,
                config=config,
                metadata={
                    "strategy": f"{strategy_name}_long",
                    **preview_metadata,
                    "mom": round(mom, 6),
                    "rsi": round(rsi, 1),
                    "atr_ratio": round(atr_ratio, 2),
                    "ret_72h": round(ret_72h, 1),
                    "sq_count": int(prev_squeeze_count),
                },
            )
        )

    return signals, last_short, last_long


@dataclass(slots=True, frozen=True)
class _SqueezeV8FeatureRow:
    rsi_14: float
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
            rsi_14=rsi,
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
        emitted, last_short, last_long = _emit_squeeze_entry_signals(
            signal_date=close_time,
            symbol=symbol,
            config=config,
            strategy_name="squeeze_v8.2",
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
                        emitted, last_short, last_long = _emit_squeeze_entry_signals(
                            signal_date=snapshot.signal_time,
                            symbol=symbol,
                            config=config,
                            strategy_name="squeeze_v8.2",
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
        return sorted(
            _apply_dynamic_sizing(signals, sizing_mode=active_config.sizing_mode),
            key=lambda signal: signal.signal_date,
        )

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
    return sorted(
        _apply_dynamic_sizing(signals, sizing_mode=active_config.sizing_mode),
        key=lambda signal: signal.signal_date,
    )


class SqueezeV8Strategy(SignalGenerator):
    """Squeeze V8.2: SHORT + LONG live signal generator.

    SHORT: squeeze breakout with negative momentum (all regimes)
           TP 3.0/SL 1.5, RSI >= 25, ATR ratio <= 1.5
    LONG:  squeeze breakout with positive momentum (bull regime only, ret_72h >= 6%)
           TP 4.0/SL 2.0, RSI <= 70, ATR ratio <= 1.5
    Max holding time: 72h (V8.1 change from 24h default)
    Default sizing: ridge_v1 dynamic sizing (V8.2 change)
    """

    def __init__(
        self,
        analysis_interval: str = "1h",
        poll_interval: str | None = None,
        sizing_mode: str = "ridge_v1",
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
        self.sizing_mode = _validate_sizing_mode(sizing_mode)
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
            f"SqueezeV8.2 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"analysis={self.analysis_interval} | poll={self.effective_poll_interval} | "
            f"sizing={self.sizing_mode} | "
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
        failures: list[str] = []
        self._states = {}
        self._last_committed_hour = {}
        for symbol in SYMBOLS:
            state = _SqueezeV8PreviewState()
            try:
                candles = self._client.fetch_klines(
                    symbol=symbol.replace("/", ""),
                    interval=self.analysis_interval,
                    start=start,
                    end=now,
                )
                for candle in candles:
                    if candle.close_time <= now:
                        state.commit(candle)
                        self._last_committed_hour[symbol] = candle.open_time
                self._states[symbol] = state
                if state.candle_count < _WARMUP_BARS:
                    failures.append(
                        f"{symbol}: only warmed {state.candle_count} candles, need {_WARMUP_BARS}"
                    )
                    continue
                if (
                    state.last_row is None
                    or np.isnan(state.last_row.mom_slope)
                    or np.isnan(state.last_row.atr_ratio)
                ):
                    failures.append(
                        f"{symbol}: warmup indicators are incomplete after {state.candle_count} candles"
                    )
                    continue
                print(
                    f"  {symbol}: warmed up {state.candle_count} bars",
                    file=sys.stderr,
                )
            except Exception as exc:
                failures.append(f"{symbol}: warmup fetch failed: {exc}")

        if failures:
            details = "\n".join(f"  - {failure}" for failure in failures)
            raise FatalSignalError(
                "SqueezeV8 warmup failed; refusing to trade without full candle history:\n"
                f"{details}"
            )

    def poll(self) -> list[Signal] | None:
        assert self._client is not None
        now = self.current_time()
        signals: list[Signal] = []

        with ThreadPoolExecutor(max_workers=6) as pool:
            future_to_symbol = {
                pool.submit(self._check_symbol, symbol, now): symbol
                for symbol in SYMBOLS
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    sigs = future.result()
                    signals.extend(sigs)
                except FatalSignalError:
                    raise
                except Exception as exc:
                    print(f"Error checking {symbol}: {exc}", file=sys.stderr)

        signals = _apply_dynamic_sizing(signals, sizing_mode=self.sizing_mode)
        return signals if signals else None

    def _check_symbol(self, symbol: str, now: datetime) -> list[Signal]:
        assert self._client is not None

        state = self._states.get(symbol)
        if state is None:
            raise FatalSignalError(f"{symbol}: warmup state is missing")

        preview_enabled = self.effective_poll_interval != self.analysis_interval

        # Hourly mode: _check_symbol_pandas fetches full lookback and commits
        # new candles itself — no need for a separate commit fetch here.
        if not preview_enabled:
            return self._check_symbol_pandas(symbol, now)

        # Preview path: commit any newly closed hourly candles since last commit
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
            raise FatalSignalError(f"{symbol}: warmup state has no committed candles")
        if state.candle_count < _WARMUP_BARS:
            raise FatalSignalError(
                f"{symbol}: only {state.candle_count} candles available, need {_WARMUP_BARS}"
            )
        if np.isnan(state.last_row.mom_slope) or np.isnan(state.last_row.atr_ratio):
            raise FatalSignalError(
                f"{symbol}: warmup indicators are incomplete after {state.candle_count} candles"
            )

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
        emitted, self._last_short[symbol], self._last_long[symbol] = _emit_squeeze_entry_signals(
            signal_date=now,
            symbol=symbol,
            config=self,
            strategy_name="squeeze_v8.2",
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
        """Hourly polling path: full pandas recompute.

        Only used when poll_interval == analysis_interval (default 1h mode).
        Also commits newly closed candles to the incremental preview state.
        """
        assert self._client is not None
        import pandas as pd

        state = self._states.get(symbol)
        if state is None:
            raise FatalSignalError(f"{symbol}: warmup state is missing")

        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval=self.analysis_interval,
            start=start,
            end=now,
        )

        if len(candles) < _WARMUP_BARS:
            raise FatalSignalError(
                f"{symbol}: live fetch returned {len(candles)} candles, need {_WARMUP_BARS}"
            )

        # Commit new closed candles to incremental state before any early returns
        last_committed = self._last_committed_hour.get(symbol)
        if last_committed is None:
            raise FatalSignalError(
                f"{symbol}: state exists but _last_committed_hour is missing"
            )
        for c in candles:
            if c.close_time <= now and c.open_time > last_committed:
                state.commit(c)
                self._last_committed_hour[symbol] = c.open_time

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
            raise FatalSignalError(
                f"{symbol}: only {last_idx + 1} closed candles available, need {_WARMUP_BARS}"
            )

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
        emitted, self._last_short[symbol], self._last_long[symbol] = _emit_squeeze_entry_signals(
            signal_date=now,
            symbol=symbol,
            config=self,
            strategy_name="squeeze_v8.2",
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
            sizing_mode=self.sizing_mode,
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
