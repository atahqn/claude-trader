"""Live signal generator: Squeeze V8.3 Strategy.

V8.3 changes over V8.2:
  - SHORT signals get quality-tiered TP/SL based on (sq_count, RSI, ATR ratio):
      Tier A (sq [10,20] AND RSI < 45):  TP 3.25% / SL 1.5%  (wider TP)
      Tier C (toxic combos):              TP 3.0%  / SL 1.1%  (tighter SL)
      Tier B (everything else):           TP 3.0%  / SL 1.5%  (unchanged)
  - LONG signals filtered when ret_72h > 25% (overextended bull regime)

Current live logic:
  1. SQUEEZE SHORT
    - BB/KC squeeze release after 7+ compressed bars
    - negative momentum, RSI >= 25, ATR ratio <= 1.5
    - TP/SL per quality tier (see above), cooldown 12h

  2. BULL PULLBACK RECLAIM LONG
    - strong bull regime (10% <= ret_72h <= 25%) with positive momentum
    - shallow pullback to BB midline / EMA20 support, then decisive reclaim
    - RSI <= 75, ATR ratio <= 1.2, bullish bar, prior impulse
    - TP/SL 4.0% / 2.0%, cooldown 12h

  Conflict resolution: SHORT priority when both fire for the same ticker+time.
  Max holding time: 72h
  Position sizing: ridge_v1 dynamic sizing with separate SHORT and LONG models

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
    "AVAX/USDT", "XRP/USDT", "LINK/USDT", "BNB/USDT",
    "ENA/USDT", "DOGE/USDT", "ALGO/USDT",
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
_RIDGE_V1_SHORT_MEAN = 0.2885689587426325
_RIDGE_V1_SHORT_STD = 0.32868843251811886
_RIDGE_V1_SHORT_ALPHA = 0.34
_RIDGE_V1_CLIP_LO = 0.45
_RIDGE_V1_CLIP_HI = 1.7
_RIDGE_V1_SCALE_2 = 0.9964448114052744

# --- Pullback LONG ridge sizing (v1, alpha=0.25) -------------------------
_PULLBACK_FEATURE_COLUMNS = (
    "rsi_14",
    "atr_14",
    "atr_ratio",
    "ret_72h",
    "ema_20",
    "_bb_ma_20",
    "bb_upper",
    "mom_slope",
)
_PULLBACK_WARMUP_BARS = required_warmup(_PULLBACK_FEATURE_COLUMNS)

# Union of squeeze and pullback columns — computed once per poll instead of twice.
_COMBINED_FEATURE_COLUMNS = tuple(dict.fromkeys(
    SQUEEZE_V8_FEATURE_COLUMNS + _PULLBACK_FEATURE_COLUMNS
))

_PULLBACK_V1_BETA = (
    0.0203650314688874,
    -0.54086920213952,
    0.00675308728303108,
    0.921725996134645,
    0.145963989406905,
    -0.146634174435791,
    0.177328696725871,
    -0.264529959823252,
    0.423949474728184,
    -0.000791759249875825,
    -0.421579168844033,
    0.0659598527789946,
    0.0,
    0.0,
)
_PULLBACK_V1_MEAN = 0.536585365853654
_PULLBACK_V1_STD = 1.00910777565916
_PULLBACK_V1_ALPHA = 0.25
_PULLBACK_V1_SCALE = 0.9989947888
_PULLBACK_CLIP_LO = 0.45
_PULLBACK_CLIP_HI = 1.70


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


def _ridge_v1_short_size_multiplier(
    signal: Signal,
    *,
    cluster_side: int,
    cluster_total: int,
) -> float:
    """Ridge sizing for squeeze SHORT signals."""
    score = float(
        np.dot(
            np.asarray(_RIDGE_V1_SHORT_BETA, dtype=float),
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
    z_score = (score - _RIDGE_V1_SHORT_MEAN) / _RIDGE_V1_SHORT_STD
    raw = float(np.clip(1.0 + _RIDGE_V1_SHORT_ALPHA * z_score, 0.2, 3.0))
    normalized = float(np.clip(raw, _RIDGE_V1_CLIP_LO, _RIDGE_V1_CLIP_HI))
    return float(
        np.clip(
            normalized / _RIDGE_V1_SCALE_2,
            _RIDGE_V1_CLIP_LO,
            _RIDGE_V1_CLIP_HI,
        )
    )


def _pullback_v1_feature_vector(
    signal: Signal,
    *,
    cluster_side: int,
    cluster_total: int,
) -> tuple[float, ...]:
    ret_72h = _signal_metadata_float(signal, "ret_72h", 15.0)
    rsi = _signal_metadata_float(signal, "rsi", 55.0)
    atr_ratio = _signal_metadata_float(signal, "atr_ratio", 0.9)
    pullback_depth = _signal_metadata_float(signal, "pullback_depth", 0.3)
    mom_slope = _signal_metadata_float(signal, "mom_slope", 0.0)
    reclaim_strength = _signal_metadata_float(signal, "reclaim_strength", 0.5)

    ret = (ret_72h - 15.0) / 5.0
    rsi_c = (rsi - 55.0) / 10.0
    atr = (atr_ratio - 0.9) / 0.15
    depth = (pullback_depth - 0.3) / 0.3
    mom = mom_slope * 1000.0
    reclaim = (reclaim_strength - 0.5) / 0.3
    side_cluster = math.log1p(max(float(cluster_side) - 1.0, 0.0))
    total_cluster = math.log1p(max(float(cluster_total) - 1.0, 0.0))

    return (
        1.0, ret, ret ** 2, rsi_c, rsi_c ** 2,
        atr, atr ** 2, depth, depth ** 2,
        mom, reclaim, reclaim ** 2,
        side_cluster, total_cluster,
    )


def _pullback_ridge_size_multiplier(
    signal: Signal,
    *,
    cluster_side: int,
    cluster_total: int,
) -> float:
    features = _pullback_v1_feature_vector(
        signal, cluster_side=cluster_side, cluster_total=cluster_total,
    )
    score = float(
        np.dot(
            np.asarray(_PULLBACK_V1_BETA, dtype=float),
            np.asarray(features, dtype=float),
        )
    )
    z_score = (score - _PULLBACK_V1_MEAN) / _PULLBACK_V1_STD
    raw = float(np.clip(1.0 + _PULLBACK_V1_ALPHA * z_score, 0.2, 3.0))
    clipped = float(np.clip(raw, _PULLBACK_CLIP_LO, _PULLBACK_CLIP_HI))
    normalized = clipped / _PULLBACK_V1_SCALE
    return float(np.clip(normalized, _PULLBACK_CLIP_LO, _PULLBACK_CLIP_HI))


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
            is_pullback = signal.metadata.get("strategy") == "pullback_long"
            if is_pullback:
                size_multiplier = _pullback_ridge_size_multiplier(
                    signal,
                    cluster_side=cluster_side,
                    cluster_total=cluster_total,
                )
            else:
                size_multiplier = _ridge_v1_short_size_multiplier(
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


def _classify_short_tier(
    sq_count: int,
    rsi: float,
    atr_ratio: float,
) -> str:
    """Classify SHORT signal quality tier (V8.3).

    Tier A: sq [10,20] AND RSI < 45 — high-quality setups, wider TP.
    Tier C: toxic feature combinations — tighter SL.
    Tier B: everything else — standard V8.2 parameters.
    """
    if sq_count < 10 and rsi >= 50:
        return "C"
    if sq_count >= 20 and atr_ratio >= 1.0:
        return "C"
    if 10 <= sq_count < 15 and atr_ratio >= 1.2:
        return "C"
    if 10 <= sq_count <= 20 and rsi < 45:
        return "A"
    return "B"


# V8.3 tier-based TP/SL defaults
_TIER_TP = {"A": 3.25, "B": 3.0, "C": 3.0}
_TIER_SL = {"A": 1.5, "B": 1.5, "C": 1.1}

# V8.3 LONG overextension cap
_LONG_RET_MAX = 25.0


@dataclass(slots=True, frozen=True)
class SqueezeV8Config:
    analysis_interval: str = "1h"
    poll_interval: str | None = None
    sizing_mode: str = "ridge_v1"
    leverage: float = 1.0
    enable_short: bool = True
    enable_pullback_long: bool = True
    short_tp: float = 3.0
    short_sl: float = 1.5
    short_cooldown_h: float = 12.0
    short_rsi_floor: float = 25.0
    long_tp: float = 4.0
    long_sl: float = 2.0
    long_cooldown_h: float = 12.0
    long_ret_max: float = _LONG_RET_MAX
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


def _build_squeeze_short_signal(
    *,
    signal_date: datetime,
    symbol: str,
    config: Any,
    metadata: dict[str, object],
    size_multiplier: float = 1.0,
) -> Signal:
    # V8.3: classify SHORT quality tier and apply per-signal TP/SL
    sq_count = int(metadata.get("sq_count", 12))
    rsi = float(metadata.get("rsi", 40.0))
    atr_ratio = float(metadata.get("atr_ratio", 0.9))
    tier = _classify_short_tier(sq_count, rsi, atr_ratio)
    tp = _TIER_TP[tier]
    sl = _TIER_SL[tier]
    metadata = {**metadata, "quality_tier": tier}

    return Signal(
        signal_date=signal_date,
        position_type=PositionType.SHORT,
        ticker=symbol,
        tp_pct=tp,
        sl_pct=sl,
        leverage=config.leverage,
        market_type=getattr(config, "market_type", MarketType.FUTURES),
        taker_fee_rate=getattr(config, "taker_fee_rate", 0.0005),
        max_holding_hours=getattr(config, "max_holding_hours", 72),
        size_multiplier=size_multiplier,
        metadata=metadata,
    )


def _emit_squeeze_short_signal(
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
    last_short: datetime | None,
    source_period_start: datetime | None = None,
    source_period_gate: SourcePeriodGate | None = None,
    gate_key: object = "default",
) -> tuple[Signal | None, datetime | None]:
    """Check squeeze SHORT conditions and emit signal if met."""
    if atr_ratio > config.atr_ratio_max:
        return None, last_short
    if prev_squeeze_count < config.min_squeeze_bars or squeeze_on:
        return None, last_short
    if mom >= 0 or rsi < config.short_rsi_floor:
        return None, last_short
    if last_short is not None and (signal_date - last_short).total_seconds() < config.short_cooldown_h * 3600:
        return None, last_short

    gate_ok = (
        source_period_start is None
        or source_period_gate is None
        or source_period_gate.allow(source_period_start, key=gate_key)
    )
    if not gate_ok:
        return None, last_short

    if source_period_start is not None and source_period_gate is not None:
        source_period_gate.mark(source_period_start, key=gate_key)

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

    signal = _build_squeeze_short_signal(
        signal_date=signal_date,
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
    return signal, signal_date


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
    candle_buffer: deque[Candle] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.tr_tail = deque(maxlen=13)
        self.atr_tail = deque(maxlen=71)
        self.close_tail_19 = deque(maxlen=19)
        self.close_tail_72 = deque(maxlen=72)
        self.candle_buffer = deque(maxlen=_LOOKBACK_BARS)

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
        self.candle_buffer.append(candle)
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


def _build_pullback_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return compute_indicator_frame(frame, _PULLBACK_FEATURE_COLUMNS)


def _build_pullback_feature_frames(
    prepared_context: PreparedMarketContext,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames[symbol] = _build_pullback_feature_frame(
            prepared_context.for_symbol(symbol).frame
        )
    return frames


def _safe_value(row: pd.Series, column: str, default: float) -> float:
    value = row.get(column, default)
    return default if pd.isna(value) else float(value)


def _safe_feature_value(value: float, default: float) -> float:
    return default if np.isnan(value) else float(value)


def _generate_squeeze_short_signals(
    frame: pd.DataFrame,
    symbol: str,
    start: datetime,
    end: datetime,
    config: SqueezeV8Config,
) -> list[Signal]:
    """Generate squeeze SHORT signals from a squeeze feature frame."""
    if frame.empty:
        return []

    signals: list[Signal] = []
    last_short: datetime | None = None

    first_eval_index = max(min(config.warmup_bars - 1, len(frame) - 1), 1)
    for index in range(first_eval_index, len(frame)):
        row = frame.iloc[index]
        prev = frame.iloc[index - 1]
        close_time = row["close_time"]
        if close_time < start or close_time >= end:
            continue

        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            continue

        rsi = _safe_value(row, "rsi_14", 50.0)
        mom = _safe_value(row, "mom_slope", 0.0)
        atr_ratio = _safe_value(row, "atr_ratio", 1.0)
        sq_count = int(_safe_value(prev, "squeeze_count", 0.0))
        signal, last_short = _emit_squeeze_short_signal(
            signal_date=close_time,
            symbol=symbol,
            config=config,
            strategy_name="squeeze_v8.3",
            prev_squeeze_count=sq_count,
            squeeze_on=bool(row.get("squeeze_on", False)),
            mom=mom,
            rsi=rsi,
            atr_ratio=atr_ratio,
            last_short=last_short,
        )
        if signal is not None:
            signals.append(signal)

    return signals


def _check_pullback_entry(
    frame: pd.DataFrame,
    index: int,
    *,
    long_ret_max: float = _LONG_RET_MAX,
) -> tuple[bool, dict[str, object]]:
    """Check if bull pullback reclaim LONG entry conditions are met."""
    regime_min = 10.0
    rsi_cap = 75.0
    atr_ratio_max = 1.2
    lookback_above_ema = 5
    reclaim_atr_min = 0.3
    min_body_ratio = 0.4
    impulse_lookback = 10

    if index < 2:
        return False, {}

    row = frame.iloc[index]

    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}

    ret_72h = _safe_value(row, "ret_72h", 0.0)
    if ret_72h < regime_min:
        return False, {}
    # V8.3: filter overextended bull regime
    if ret_72h > long_ret_max:
        return False, {}

    rsi = _safe_value(row, "rsi_14", 50.0)
    if rsi > rsi_cap:
        return False, {}

    atr_ratio = _safe_value(row, "atr_ratio", 1.0)
    if atr_ratio > atr_ratio_max:
        return False, {}

    mom_slope = _safe_value(row, "mom_slope", 0.0)
    if mom_slope <= 0:
        return False, {}

    bb_mid = _safe_value(row, "_bb_ma_20", 0.0)
    ema20 = float(row["ema_20"])
    support = bb_mid if bb_mid > 0 else ema20
    atr = _safe_value(row, "atr_14", 0.0)
    if atr <= 0:
        return False, {}
    close = float(row["close"])
    open_price = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])

    # Strong reclaim: close above support + 0.3 * ATR
    if close <= support + reclaim_atr_min * atr:
        return False, {}

    # Bullish bar with minimum body ratio
    bar_range = high - low
    if bar_range <= 0:
        return False, {}
    body = close - open_price
    if body <= 0:
        return False, {}
    if body / bar_range < min_body_ratio:
        return False, {}

    # Pullback: prev bar's low at or below support
    prev = frame.iloc[index - 1]
    prev_low = float(prev["low"])
    prev_support = _safe_value(prev, "_bb_ma_20", _safe_value(prev, "ema_20", support))
    if prev_low > prev_support:
        return False, {}

    # Shallow pullback: prev close not too far below support
    prev_close = float(prev["close"])
    if prev_close < prev_support - 1.0 * atr:
        return False, {}

    # Price was above support recently (uptrend confirmation)
    lookback_start = max(0, index - lookback_above_ema - 1)
    lookback_end = index - 1
    if lookback_end <= lookback_start:
        return False, {}
    above_count = 0
    for lb_idx in range(lookback_start, lookback_end):
        lb_row = frame.iloc[lb_idx]
        lb_support = _safe_value(lb_row, "_bb_ma_20", _safe_value(lb_row, "ema_20", 0.0))
        if float(lb_row["close"]) > lb_support:
            above_count += 1
    if above_count < (lookback_end - lookback_start) / 2:
        return False, {}

    # Prior impulse: recent bar touched BB upper band
    impulse_start = max(0, index - impulse_lookback)
    impulse_end = index - 1
    had_impulse = False
    for imp_idx in range(impulse_start, impulse_end):
        imp_row = frame.iloc[imp_idx]
        imp_bb_upper = _safe_value(imp_row, "bb_upper", float("inf"))
        if float(imp_row["high"]) >= imp_bb_upper:
            had_impulse = True
            break
    if not had_impulse:
        return False, {}

    reclaim_strength = (close - support) / atr if atr > 0 else 0.0
    pullback_depth = (prev_support - prev_low) / atr if atr > 0 else 0.0

    metadata: dict[str, object] = {
        "strategy": "pullback_long",
        "ret_72h": round(ret_72h, 2),
        "rsi": round(rsi, 2),
        "atr_ratio": round(atr_ratio, 3),
        "pullback_depth": round(pullback_depth, 3),
        "mom_slope": round(mom_slope, 6),
        "reclaim_strength": round(reclaim_strength, 3),
    }
    return True, metadata


def _generate_pullback_long_signals(
    frame: pd.DataFrame,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    long_tp: float = 4.0,
    long_sl: float = 2.0,
    long_cooldown_h: float = 12.0,
    long_ret_max: float = _LONG_RET_MAX,
    max_holding_hours: int = 72,
    leverage: float = 1.0,
    market_type: MarketType = MarketType.FUTURES,
    taker_fee_rate: float = 0.0005,
) -> list[Signal]:
    """Generate bull pullback reclaim LONG signals from a pullback feature frame."""
    if frame.empty or len(frame) < _PULLBACK_WARMUP_BARS + 2:
        return []

    signals: list[Signal] = []
    last_signal_time: datetime | None = None

    first_eval_index = max(_PULLBACK_WARMUP_BARS, 2)
    for index in range(first_eval_index, len(frame)):
        close_time = frame.iloc[index]["close_time"]
        if close_time < start or close_time >= end:
            continue

        ok, metadata = _check_pullback_entry(frame, index, long_ret_max=long_ret_max)
        if not ok:
            continue

        # Cooldown
        if last_signal_time is not None:
            hours_since = (close_time - last_signal_time).total_seconds() / 3600.0
            if hours_since < long_cooldown_h:
                continue

        signal = Signal(
            signal_date=close_time,
            position_type=PositionType.LONG,
            ticker=symbol,
            tp_pct=long_tp,
            sl_pct=long_sl,
            leverage=leverage,
            market_type=market_type,
            taker_fee_rate=taker_fee_rate,
            max_holding_hours=max_holding_hours,
            metadata=metadata,
        )
        signals.append(signal)
        last_signal_time = close_time

    return signals


def _resolve_short_long_conflicts(signals: list[Signal]) -> list[Signal]:
    """If SHORT and LONG fire for same ticker at same time, SHORT wins."""
    short_keys: set[tuple[datetime, str]] = set()
    for s in signals:
        if s.position_type is PositionType.SHORT:
            short_keys.add((s.signal_date, s.ticker))
    return [
        s for s in signals
        if s.position_type is PositionType.SHORT
        or (s.signal_date, s.ticker) not in short_keys
    ]


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
                rsi = _safe_feature_value(row.rsi_14, 50.0)
                mom = _safe_feature_value(row.mom_slope, 0.0)
                atr_ratio = _safe_feature_value(row.atr_ratio, 1.0)
                sq_count = int(prev.squeeze_count)
                signal, last_short = _emit_squeeze_short_signal(
                    signal_date=snapshot.signal_time,
                    symbol=symbol,
                    config=config,
                    strategy_name="squeeze_v8.3",
                    prev_squeeze_count=sq_count,
                    squeeze_on=row.squeeze_on,
                    mom=mom,
                    rsi=rsi,
                    atr_ratio=atr_ratio,
                    last_short=last_short,
                    source_period_start=snapshot.source_period_start,
                    source_period_gate=gate,
                )
                if signal is not None:
                    signals.append(signal)

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
        # Squeeze SHORT via preview path
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
    else:
        frames = feature_frames or build_squeeze_v8_feature_frames(
            prepared_context,
            symbols=symbols,
        )
        selected_symbols = symbols or list(frames)

        # Squeeze SHORT signals
        signals: list[Signal] = []
        for symbol in selected_symbols:
            frame = frames.get(symbol)
            if frame is None:
                continue
            signals.extend(
                _generate_squeeze_short_signals(
                    frame, symbol, active_start, active_end, active_config,
                )
            )

    # Pullback LONG signals
    if active_config.enable_pullback_long:
        available_symbols = [
            s for s in selected_symbols if s in prepared_context.symbols
        ]
        pullback_frames = _build_pullback_feature_frames(
            prepared_context, available_symbols,
        ) if available_symbols else {}
        for symbol in selected_symbols:
            pf = pullback_frames.get(symbol)
            if pf is None or pf.empty:
                continue
            signals.extend(
                _generate_pullback_long_signals(
                    pf, symbol, active_start, active_end,
                    long_tp=active_config.long_tp,
                    long_sl=active_config.long_sl,
                    long_cooldown_h=active_config.long_cooldown_h,
                    long_ret_max=active_config.long_ret_max,
                    max_holding_hours=active_config.max_holding_hours,
                    leverage=active_config.leverage,
                    market_type=active_config.market_type,
                    taker_fee_rate=active_config.taker_fee_rate,
                )
            )

    signals = _resolve_short_long_conflicts(signals)
    return sorted(
        _apply_dynamic_sizing(signals, sizing_mode=active_config.sizing_mode),
        key=lambda signal: signal.signal_date,
    )


class SqueezeV8Strategy(SignalGenerator):
    """Squeeze V8.3: SHORT + pullback LONG live signal generator.

    SHORT: squeeze breakout with negative momentum (all regimes)
           TP/SL per quality tier:
             Tier A (sq [10,20] AND RSI < 45):  TP 3.25% / SL 1.5%
             Tier C (toxic combos):              TP 3.0%  / SL 1.1%
             Tier B (default):                   TP 3.0%  / SL 1.5%
           RSI >= 25, ATR ratio <= 1.5
    LONG:  bull pullback reclaim
           TP 4.0/SL 2.0, 10% <= ret_72h <= 25%, RSI <= 75, ATR ratio <= 1.2
    Max holding time: 72h
    Default sizing: ridge_v1 dynamic sizing (separate models per side)
    """

    @property
    def symbols(self) -> list[str]:
        return list(SYMBOLS)

    def __init__(
        self,
        analysis_interval: str = "1h",
        poll_interval: str | None = None,
        sizing_mode: str = "ridge_v1",
        leverage: float = 1.0,
        # SHORT params
        short_tp: float = 3.0,
        short_sl: float = 1.5,
        short_cooldown_h: float = 12.0,
        short_rsi_floor: float = 25.0,
        # LONG params
        long_tp: float = 4.0,
        long_sl: float = 2.0,
        long_cooldown_h: float = 12.0,
        long_ret_max: float = _LONG_RET_MAX,
        enable_pullback_long: bool = True,
        # Shared params
        min_squeeze_bars: int = 7,
        atr_ratio_max: float = 1.5,
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
        self.enable_pullback_long = enable_pullback_long
        self.leverage = leverage
        self.short_tp = short_tp
        self.short_sl = short_sl
        self.short_cooldown_h = short_cooldown_h
        self.short_rsi_floor = short_rsi_floor
        self.long_tp = long_tp
        self.long_sl = long_sl
        self.long_cooldown_h = long_cooldown_h
        self.long_ret_max = long_ret_max
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
            f"SqueezeV8.3 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"analysis={self.analysis_interval} | poll={self.effective_poll_interval} | "
            f"sizing={self.sizing_mode} | "
            f"SHORT TP/SL=tiered(A:{_TIER_TP['A']}/{_TIER_SL['A']}% "
            f"B:{_TIER_TP['B']}/{_TIER_SL['B']}% C:{_TIER_TP['C']}/{_TIER_SL['C']}%) "
            f"CD={self.short_cooldown_h}h RSI>={self.short_rsi_floor} | "
            f"LONG(pullback) TP/SL={self.long_tp}/{self.long_sl}% "
            f"ret_max={self.long_ret_max}% "
            f"CD={self.long_cooldown_h}h enabled={self.enable_pullback_long} | "
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

        emitted: list[Signal] = []

        # Squeeze SHORT check
        if not np.isnan(row.mom_slope) and not np.isnan(row.atr_ratio):
            rsi = _safe_feature_value(row.rsi_14, 50.0)
            mom = _safe_feature_value(row.mom_slope, 0.0)
            atr_ratio = _safe_feature_value(row.atr_ratio, 1.0)

            source_hour_start = eval_candle.open_time
            short_sig, self._last_short[symbol] = _emit_squeeze_short_signal(
                signal_date=now,
                symbol=symbol,
                config=self,
                strategy_name="squeeze_v8.3",
                prev_squeeze_count=int(prev.squeeze_count),
                squeeze_on=row.squeeze_on,
                mom=mom,
                rsi=rsi,
                atr_ratio=atr_ratio,
                last_short=self._last_short.get(symbol),
                source_period_start=source_hour_start,
                source_period_gate=self._source_period_gate,
                gate_key=symbol,
            )
            if short_sig is not None:
                emitted.append(short_sig)

        # Pullback LONG check (full pandas path for preview mode)
        had_short = len(emitted) > 0
        if self.enable_pullback_long and not had_short:
            pb_candles = list(state.candle_buffer)
            if len(pb_candles) >= _PULLBACK_WARMUP_BARS:
                import pandas as pd
                pb_rows = [{
                    "open_time": c.open_time, "close_time": c.close_time,
                    "open": c.open, "high": c.high, "low": c.low,
                    "close": c.close, "volume": c.volume,
                } for c in pb_candles]
                df_raw = pd.DataFrame(pb_rows).sort_values("open_time").reset_index(drop=True)
                df_feat = compute_indicator_frame(df_raw, _COMBINED_FEATURE_COLUMNS)
                pullback_sigs = self._check_pullback_long(df_feat, symbol, now)
                emitted.extend(pullback_sigs)

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
        df_raw = pd.DataFrame(rows)
        if df_raw.empty:
            return []
        df_raw = df_raw.sort_values("open_time").reset_index(drop=True)

        # --- Combined feature frame (squeeze + pullback in one pass) ---
        df_feat = compute_indicator_frame(df_raw, _COMBINED_FEATURE_COLUMNS)
        last_idx = len(df_feat) - 1
        if df_feat.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx + 1 < _WARMUP_BARS:
            raise FatalSignalError(
                f"{symbol}: only {last_idx + 1} closed candles available, need {_WARMUP_BARS}"
            )

        emitted: list[Signal] = []
        row = df_feat.iloc[last_idx]
        prev = df_feat.iloc[last_idx - 1]

        if not pd.isna(row.get("mom_slope")) and not pd.isna(row.get("atr_ratio")):
            rsi = row.get("rsi_14", 50)
            if pd.isna(rsi):
                rsi = 50
            mom = row["mom_slope"]
            atr_ratio = row.get("atr_ratio", 1.0)
            if pd.isna(atr_ratio):
                atr_ratio = 1.0
            short_sig, self._last_short[symbol] = _emit_squeeze_short_signal(
                signal_date=now,
                symbol=symbol,
                config=self,
                strategy_name="squeeze_v8.3",
                prev_squeeze_count=int(prev["squeeze_count"]),
                squeeze_on=bool(row["squeeze_on"]),
                mom=float(mom),
                rsi=float(rsi),
                atr_ratio=float(atr_ratio),
                last_short=self._last_short.get(symbol),
            )
            if short_sig is not None:
                emitted.append(short_sig)

        # --- Pullback LONG check ---
        had_short = len(emitted) > 0
        if self.enable_pullback_long and not had_short:
            pullback_sigs = self._check_pullback_long(df_feat, symbol, now)
            emitted.extend(pullback_sigs)

        return emitted

    def _check_pullback_long(
        self,
        df_feat: pd.DataFrame,
        symbol: str,
        now: datetime,
    ) -> list[Signal]:
        """Check pullback LONG conditions on the last closed candle."""
        df_pullback = df_feat
        last_idx = len(df_pullback) - 1
        if df_pullback.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx < _PULLBACK_WARMUP_BARS:
            return []

        ok, metadata = _check_pullback_entry(
            df_pullback, last_idx, long_ret_max=self.long_ret_max,
        )
        if not ok:
            return []

        # Cooldown
        last_long = self._last_long.get(symbol)
        if last_long is not None:
            hours_since = (now - last_long).total_seconds() / 3600.0
            if hours_since < self.long_cooldown_h:
                return []

        self._last_long[symbol] = now

        return [Signal(
            signal_date=now,
            position_type=PositionType.LONG,
            ticker=symbol,
            tp_pct=self.long_tp,
            sl_pct=self.long_sl,
            leverage=self.leverage,
            market_type=MarketType.FUTURES,
            taker_fee_rate=0.0005,
            max_holding_hours=self.max_holding_hours,
            metadata=metadata,
        )]

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
            enable_pullback_long=self.enable_pullback_long,
            short_tp=self.short_tp,
            short_sl=self.short_sl,
            short_cooldown_h=self.short_cooldown_h,
            short_rsi_floor=self.short_rsi_floor,
            long_tp=self.long_tp,
            long_sl=self.long_sl,
            long_cooldown_h=self.long_cooldown_h,
            long_ret_max=self.long_ret_max,
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
