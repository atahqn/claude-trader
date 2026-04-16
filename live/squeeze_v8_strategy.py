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
  - the reference calibration is 1h, but lower analysis intervals are supported
  - poll interval defaults to the analysis interval, preserving close-only behavior
  - when poll interval is lower than the analysis interval, the strategy evaluates
    the current in-progress analysis candle snapshot at those poll boundaries
  - uses incremental _SqueezeV8PreviewState per symbol for O(1) indicator updates

Look-ahead bias prevention:
  - close-only mode uses the last fully closed analysis candle
  - preview mode uses the currently visible analysis candle snapshot
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
    compute_rsi_from_ewm_means,
    linear_regression_slope,
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
_BASE_ANALYSIS_INTERVAL = "1h"
_BASE_LOOKBACK_BARS = 100
_BASE_RSI_BARS = 14
_BASE_ATR_BARS = 14
_BASE_ATR_AVG_BARS = 72
_BASE_RETURN_BARS = 72
_BASE_BB_BARS = 20
_BASE_MOM_BARS = 20
_BASE_PULLBACK_ABOVE_SUPPORT_BARS = 5
_BASE_PULLBACK_IMPULSE_BARS = 10
_SUPPORTED_POLL_INTERVALS = {"1h", "30m", "15m", "5m", "1m"}
_SUPPORTED_ANALYSIS_INTERVALS = _SUPPORTED_POLL_INTERVALS
_SUPPORTED_SIZING_MODES = {"baseline", "ridge_v1"}

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


def _bars_for_same_duration(base_bars: int, analysis_interval: str) -> int:
    analysis_seconds = interval_to_seconds(analysis_interval)
    base_seconds = interval_to_seconds(_BASE_ANALYSIS_INTERVAL)
    if analysis_seconds <= 0 or analysis_seconds > base_seconds:
        raise ValueError("analysis_interval must be less than or equal to 1h")
    if base_seconds % analysis_seconds != 0:
        raise ValueError("analysis_interval must divide 1h exactly")
    return max(1, base_bars * (base_seconds // analysis_seconds))


def _compute_squeeze_count(values: pd.Series) -> pd.Series:
    counts: list[int] = []
    count = 0
    for active in values.fillna(False):
        count = count + 1 if bool(active) else 0
        counts.append(count)
    return pd.Series(counts, index=values.index, dtype="int64")


@dataclass(slots=True, frozen=True)
class SqueezeV8Periods:
    analysis_interval: str
    bars_per_hour: int
    rsi_bars: int
    atr_bars: int
    atr_avg_bars: int
    return_bars: int
    bb_bars: int
    mom_bars: int
    pullback_above_support_bars: int
    pullback_impulse_bars: int
    min_squeeze_bars: int
    warmup_bars: int
    lookback_bars: int
    rsi_alpha: float
    ema_alpha: float

    @staticmethod
    def for_interval(
        analysis_interval: str,
        *,
        min_squeeze_bars: int,
        lookback_bars: int = _BASE_LOOKBACK_BARS,
        warmup_bars: int | None = None,
    ) -> "SqueezeV8Periods":
        bars_per_hour = _bars_for_same_duration(1, analysis_interval)
        rsi_bars = _bars_for_same_duration(_BASE_RSI_BARS, analysis_interval)
        atr_bars = _bars_for_same_duration(_BASE_ATR_BARS, analysis_interval)
        atr_avg_bars = _bars_for_same_duration(_BASE_ATR_AVG_BARS, analysis_interval)
        return_bars = _bars_for_same_duration(_BASE_RETURN_BARS, analysis_interval)
        bb_bars = _bars_for_same_duration(_BASE_BB_BARS, analysis_interval)
        mom_bars = _bars_for_same_duration(_BASE_MOM_BARS, analysis_interval)
        scaled_min_squeeze_bars = _bars_for_same_duration(min_squeeze_bars, analysis_interval)
        computed_warmup = max(
            rsi_bars,
            atr_bars + atr_avg_bars - 1,
            return_bars + 1,
            bb_bars,
            mom_bars,
        )
        final_warmup = computed_warmup if warmup_bars is None else max(warmup_bars, computed_warmup)
        scaled_lookback = max(
            _bars_for_same_duration(lookback_bars, analysis_interval),
            final_warmup + bars_per_hour,
        )
        return SqueezeV8Periods(
            analysis_interval=analysis_interval,
            bars_per_hour=bars_per_hour,
            rsi_bars=rsi_bars,
            atr_bars=atr_bars,
            atr_avg_bars=atr_avg_bars,
            return_bars=return_bars,
            bb_bars=bb_bars,
            mom_bars=mom_bars,
            pullback_above_support_bars=_bars_for_same_duration(
                _BASE_PULLBACK_ABOVE_SUPPORT_BARS,
                analysis_interval,
            ),
            pullback_impulse_bars=_bars_for_same_duration(
                _BASE_PULLBACK_IMPULSE_BARS,
                analysis_interval,
            ),
            min_squeeze_bars=scaled_min_squeeze_bars,
            warmup_bars=final_warmup,
            lookback_bars=scaled_lookback,
            rsi_alpha=1.0 / rsi_bars,
            ema_alpha=2.0 / (bb_bars + 1.0),
        )


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
    warmup_bars: int | None = None

    def __post_init__(self) -> None:
        if self.analysis_interval not in _SUPPORTED_ANALYSIS_INTERVALS:
            supported = ", ".join(sorted(_SUPPORTED_ANALYSIS_INTERVALS))
            raise ValueError(f"SqueezeV8 analysis_interval must be one of: {supported}")
        _validate_sizing_mode(self.sizing_mode)
        if self.analysis_interval != "1h" and self.sizing_mode == "ridge_v1":
            raise ValueError(
                "ridge_v1 sizing is calibrated only for 1h analysis_interval; use baseline for lower timeframes"
            )
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

    @property
    def periods(self) -> SqueezeV8Periods:
        return SqueezeV8Periods.for_interval(
            self.analysis_interval,
            min_squeeze_bars=self.min_squeeze_bars,
            lookback_bars=_BASE_LOOKBACK_BARS,
            warmup_bars=self.warmup_bars,
        )


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
    periods = _config_periods(config)
    if atr_ratio > config.atr_ratio_max:
        return None, last_short
    if prev_squeeze_count < periods.min_squeeze_bars or squeeze_on:
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
            "source_period_start": source_period_start.isoformat(),
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
            "sq_count": round(prev_squeeze_count / periods.bars_per_hour, 3),
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
    periods: SqueezeV8Periods
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
    close_tail_bb: deque[float] = None  # type: ignore[assignment]
    close_tail_ret: deque[float] = None  # type: ignore[assignment]
    last_row: _SqueezeV8FeatureRow | None = None
    candle_buffer: deque[Candle] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.tr_tail = deque(maxlen=max(self.periods.atr_bars - 1, 1))
        self.atr_tail = deque(maxlen=max(self.periods.atr_avg_bars - 1, 1))
        self.close_tail_bb = deque(maxlen=max(self.periods.bb_bars - 1, 1))
        self.close_tail_ret = deque(maxlen=max(self.periods.return_bars, 1))
        self.candle_buffer = deque(maxlen=max(self.periods.lookback_bars, 1))

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
                alpha=self.periods.rsi_alpha,
                min_periods=self.periods.rsi_bars,
            )
            loss_mean, loss_num, loss_den, loss_count = _ewm_next(
                self.loss_num,
                self.loss_den,
                self.loss_count,
                max(-delta, 0.0),
                alpha=self.periods.rsi_alpha,
                min_periods=self.periods.rsi_bars,
            )

        rsi = compute_rsi_from_ewm_means(gain_mean, loss_mean)

        tr_value = true_range_value(candle.high, candle.low, prev_close)
        atr_14 = np.nan
        if len(self.tr_tail) == self.periods.atr_bars - 1:
            atr_14 = (sum(self.tr_tail) + tr_value) / float(self.periods.atr_bars)

        atr_72_avg = np.nan
        if not np.isnan(atr_14) and len(self.atr_tail) == self.periods.atr_avg_bars - 1:
            atr_72_avg = (sum(self.atr_tail) + atr_14) / float(self.periods.atr_avg_bars)

        atr_ratio = np.nan
        if not np.isnan(atr_14) and not np.isnan(atr_72_avg) and atr_72_avg != 0.0:
            atr_ratio = atr_14 / atr_72_avg

        ret_72h = np.nan
        if len(self.close_tail_ret) == self.periods.return_bars and self.close_tail_ret[0] != 0.0:
            ret_72h = (close / self.close_tail_ret[0] - 1.0) * 100.0

        bb_upper = np.nan
        bb_lower = np.nan
        mom_slope = np.nan
        if len(self.close_tail_bb) == self.periods.bb_bars - 1:
            closes_for_window = list(self.close_tail_bb) + [close]
            bb_mean = float(np.mean(closes_for_window))
            bb_std = float(np.std(closes_for_window, ddof=1))
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            mom_slope = linear_regression_slope(closes_for_window)

        ema_20, ema_num, ema_den, ema_count = _ewm_next(
            self.ema_num,
            self.ema_den,
            self.ema_count,
            close,
            alpha=self.periods.ema_alpha,
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
        self.close_tail_bb.append(step.close)
        self.close_tail_ret.append(step.close)
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


def _build_strategy_feature_frame(
    frame: pd.DataFrame,
    periods: SqueezeV8Periods,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(frame.columns):
        return frame.copy()

    df = frame.copy()
    delta_close = df["close"].diff()
    gain = delta_close.clip(lower=0)
    loss = (-delta_close).clip(lower=0)
    gain_ewm = gain.ewm(alpha=periods.rsi_alpha, min_periods=periods.rsi_bars).mean()
    loss_ewm = loss.ewm(alpha=periods.rsi_alpha, min_periods=periods.rsi_bars).mean()
    df["rsi_14"] = compute_rsi_from_ewm_means(gain_ewm, loss_ewm)

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(periods.atr_bars).mean()
    df["atr_72_avg"] = df["atr_14"].rolling(periods.atr_avg_bars).mean()
    df["atr_ratio"] = df["atr_14"] / df["atr_72_avg"]

    df["ret_72h"] = df["close"].pct_change(periods.return_bars) * 100.0
    df["_bb_ma_20"] = df["close"].rolling(periods.bb_bars).mean()
    df["_bb_std_20"] = df["close"].rolling(periods.bb_bars).std()
    df["bb_upper"] = df["_bb_ma_20"] + 2.0 * df["_bb_std_20"]
    df["bb_lower"] = df["_bb_ma_20"] - 2.0 * df["_bb_std_20"]
    df["ema_20"] = df["close"].ewm(span=periods.bb_bars).mean()
    df["kc_upper"] = df["ema_20"] + 1.5 * df["atr_14"]
    df["kc_lower"] = df["ema_20"] - 1.5 * df["atr_14"]
    df["squeeze_on"] = (
        (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    )
    df["squeeze_count"] = _compute_squeeze_count(df["squeeze_on"])
    df["mom_slope"] = df["close"].rolling(periods.mom_bars).apply(
        linear_regression_slope,
        raw=True,
    )
    return df


def build_squeeze_v8_feature_frame(
    frame: pd.DataFrame,
    *,
    analysis_interval: str = "1h",
    min_squeeze_bars: int = 7,
    warmup_bars: int | None = None,
) -> pd.DataFrame:
    periods = SqueezeV8Periods.for_interval(
        analysis_interval,
        min_squeeze_bars=min_squeeze_bars,
        warmup_bars=warmup_bars,
    )
    return _build_strategy_feature_frame(frame, periods)


def build_squeeze_v8_feature_frames(
    prepared_context: PreparedMarketContext,
    *,
    symbols: list[str] | None = None,
    analysis_interval: str = "1h",
    min_squeeze_bars: int = 7,
    warmup_bars: int | None = None,
) -> dict[str, pd.DataFrame]:
    feature_frames: dict[str, pd.DataFrame] = {}
    selected_symbols = symbols or prepared_context.symbols
    for symbol in selected_symbols:
        feature_frames[symbol] = build_squeeze_v8_feature_frame(
            prepared_context.for_symbol(symbol).frame,
            analysis_interval=analysis_interval,
            min_squeeze_bars=min_squeeze_bars,
            warmup_bars=warmup_bars,
        )
    return feature_frames


def _build_pullback_feature_frame(
    frame: pd.DataFrame,
    *,
    analysis_interval: str = "1h",
    min_squeeze_bars: int = 7,
    warmup_bars: int | None = None,
) -> pd.DataFrame:
    periods = SqueezeV8Periods.for_interval(
        analysis_interval,
        min_squeeze_bars=min_squeeze_bars,
        warmup_bars=warmup_bars,
    )
    return _build_strategy_feature_frame(frame, periods)


def _build_pullback_feature_frames(
    prepared_context: PreparedMarketContext,
    symbols: list[str],
    *,
    analysis_interval: str = "1h",
    min_squeeze_bars: int = 7,
    warmup_bars: int | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frames[symbol] = _build_pullback_feature_frame(
            prepared_context.for_symbol(symbol).frame,
            analysis_interval=analysis_interval,
            min_squeeze_bars=min_squeeze_bars,
            warmup_bars=warmup_bars,
        )
    return frames


def _safe_value(row: pd.Series, column: str, default: float) -> float:
    value = row.get(column, default)
    return default if pd.isna(value) else float(value)


def _safe_feature_value(value: float, default: float) -> float:
    return default if np.isnan(value) else float(value)


def _config_periods(config: Any) -> SqueezeV8Periods:
    periods = getattr(config, "periods", None)
    if periods is not None:
        return periods
    return SqueezeV8Periods.for_interval(
        getattr(config, "analysis_interval", "1h"),
        min_squeeze_bars=getattr(config, "min_squeeze_bars", 7),
        warmup_bars=getattr(config, "warmup_bars", None),
    )


def _float_array(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return frame[column].to_numpy(dtype=float, copy=False)


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
    periods = config.periods

    close_times = frame["close_time"].tolist()
    mom_values = _float_array(frame, "mom_slope")
    atr_ratio_values = _float_array(frame, "atr_ratio")
    rsi_values = _float_array(frame, "rsi_14")
    squeeze_count_values = _float_array(frame, "squeeze_count", 0.0)
    if "squeeze_on" in frame.columns:
        squeeze_on_values = frame["squeeze_on"].fillna(False).to_numpy(dtype=bool, copy=False)
    else:
        squeeze_on_values = np.zeros(len(frame), dtype=bool)

    first_eval_index = max(min(periods.warmup_bars - 1, len(frame) - 1), 1)
    for index in range(first_eval_index, len(frame)):
        close_time = close_times[index]
        if close_time < start or close_time >= end:
            continue

        mom_raw = mom_values[index]
        atr_ratio_raw = atr_ratio_values[index]
        if np.isnan(mom_raw) or np.isnan(atr_ratio_raw):
            continue

        rsi_raw = rsi_values[index]
        prev_sq_count_raw = squeeze_count_values[index - 1]
        rsi = 50.0 if np.isnan(rsi_raw) else float(rsi_raw)
        mom = float(mom_raw)
        atr_ratio = float(atr_ratio_raw)
        sq_count = 0 if np.isnan(prev_sq_count_raw) else int(prev_sq_count_raw)
        signal, last_short = _emit_squeeze_short_signal(
            signal_date=close_time,
            symbol=symbol,
            config=config,
            strategy_name="squeeze_v8.3",
            prev_squeeze_count=sq_count,
            squeeze_on=bool(squeeze_on_values[index]),
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
    periods: SqueezeV8Periods,
    *,
    long_ret_max: float = _LONG_RET_MAX,
) -> tuple[bool, dict[str, object]]:
    """Check if bull pullback reclaim LONG entry conditions are met."""
    regime_min = 10.0
    rsi_cap = 75.0
    atr_ratio_max = 1.2
    lookback_above_ema = periods.pullback_above_support_bars
    reclaim_atr_min = 0.3
    min_body_ratio = 0.4
    impulse_lookback = periods.pullback_impulse_bars

    if index < 2:
        return False, {}

    cache = frame.attrs.get("_pullback_array_cache")
    if cache is None:
        bb_mid = _float_array(frame, "_bb_ma_20")
        ema20 = _float_array(frame, "ema_20")
        support = np.where(~np.isnan(bb_mid) & (bb_mid > 0), bb_mid, ema20)
        close = _float_array(frame, "close")
        high = _float_array(frame, "high")
        above_support = np.where(np.isnan(support), False, close > support).astype(np.int32)
        impulse_hits = np.where(np.isnan(_float_array(frame, "bb_upper")), False, high >= _float_array(frame, "bb_upper")).astype(np.int32)
        cache = {
            "ret_72h": _float_array(frame, "ret_72h"),
            "rsi_14": _float_array(frame, "rsi_14"),
            "atr_ratio": _float_array(frame, "atr_ratio"),
            "mom_slope": _float_array(frame, "mom_slope"),
            "bb_mid": bb_mid,
            "ema_20": ema20,
            "support": support,
            "atr_14": _float_array(frame, "atr_14"),
            "close": close,
            "open": _float_array(frame, "open"),
            "high": high,
            "low": _float_array(frame, "low"),
            "bb_upper": _float_array(frame, "bb_upper"),
            "close_time": frame["close_time"].tolist(),
            "above_support_prefix": np.concatenate(([0], np.cumsum(above_support))),
            "impulse_prefix": np.concatenate(([0], np.cumsum(impulse_hits))),
        }
        frame.attrs["_pullback_array_cache"] = cache

    ema20_values = cache["ema_20"]
    atr_values = cache["atr_14"]
    if np.isnan(ema20_values[index]) or np.isnan(atr_values[index]):
        return False, {}

    ret_raw = cache["ret_72h"][index]
    ret_72h = 0.0 if np.isnan(ret_raw) else float(ret_raw)
    if ret_72h < regime_min:
        return False, {}
    # V8.3: filter overextended bull regime
    if ret_72h > long_ret_max:
        return False, {}

    rsi_raw = cache["rsi_14"][index]
    rsi = 50.0 if np.isnan(rsi_raw) else float(rsi_raw)
    if rsi > rsi_cap:
        return False, {}

    atr_ratio_raw = cache["atr_ratio"][index]
    atr_ratio = 1.0 if np.isnan(atr_ratio_raw) else float(atr_ratio_raw)
    if atr_ratio > atr_ratio_max:
        return False, {}

    mom_raw = cache["mom_slope"][index]
    mom_slope = 0.0 if np.isnan(mom_raw) else float(mom_raw)
    if mom_slope <= 0:
        return False, {}

    bb_mid_raw = cache["bb_mid"][index]
    bb_mid = 0.0 if np.isnan(bb_mid_raw) else float(bb_mid_raw)
    ema20 = float(ema20_values[index])
    support = bb_mid if bb_mid > 0 else ema20
    atr = float(atr_values[index])
    if atr <= 0:
        return False, {}
    close = float(cache["close"][index])
    open_price = float(cache["open"][index])
    high = float(cache["high"][index])
    low = float(cache["low"][index])

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
    prev_low = float(cache["low"][index - 1])
    prev_bb_mid_raw = cache["bb_mid"][index - 1]
    prev_ema_raw = cache["ema_20"][index - 1]
    if not np.isnan(prev_bb_mid_raw):
        prev_support = float(prev_bb_mid_raw)
    elif not np.isnan(prev_ema_raw):
        prev_support = float(prev_ema_raw)
    else:
        prev_support = support
    if prev_low > prev_support:
        return False, {}

    # Shallow pullback: prev close not too far below support
    prev_close = float(cache["close"][index - 1])
    if prev_close < prev_support - 1.0 * atr:
        return False, {}

    # Price was above support recently (uptrend confirmation)
    lookback_start = max(0, index - lookback_above_ema - 1)
    lookback_end = index - 1
    if lookback_end <= lookback_start:
        return False, {}
    above_prefix = cache["above_support_prefix"]
    above_count = int(above_prefix[lookback_end] - above_prefix[lookback_start])
    if above_count < (lookback_end - lookback_start) / 2:
        return False, {}

    # Prior impulse: recent bar touched BB upper band
    impulse_start = max(0, index - impulse_lookback)
    impulse_end = index - 1
    impulse_prefix = cache["impulse_prefix"]
    had_impulse = impulse_end > impulse_start and (
        int(impulse_prefix[impulse_end] - impulse_prefix[impulse_start]) > 0
    )
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
    periods: SqueezeV8Periods,
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
    if frame.empty or len(frame) < periods.warmup_bars + 2:
        return []

    signals: list[Signal] = []
    last_signal_time: datetime | None = None

    first_eval_index = max(periods.warmup_bars, 2)
    close_times = frame["close_time"].tolist()
    for index in range(first_eval_index, len(frame)):
        close_time = close_times[index]
        if close_time < start or close_time >= end:
            continue

        ok, metadata = _check_pullback_entry(
            frame,
            index,
            periods,
            long_ret_max=long_ret_max,
        )
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


# ---------------------------------------------------------------------------
# Vectorized backtest-path scanners (replace per-row iloc loops)
# ---------------------------------------------------------------------------


def _generate_squeeze_short_signals_vec(
    frame: pd.DataFrame,
    symbol: str,
    start: datetime,
    end: datetime,
    config: SqueezeV8Config,
) -> list[Signal]:
    """Vectorized squeeze SHORT signal generation for backtesting.

    Produces identical results to ``_generate_squeeze_short_signals`` but
    replaces the per-row Python loop with vectorized boolean masks.  Only
    the cooldown pass and signal construction iterate — over the tiny
    subset of rows (~1-2%) that pass all gates.
    """
    if frame.empty:
        return []

    prev_sq = frame["squeeze_count"].shift(1)
    mom = frame["mom_slope"]
    atr_r = frame["atr_ratio"]
    rsi = frame["rsi_14"].fillna(50.0)

    candidate = (
        mom.notna() & atr_r.notna()
        & atr_r.le(config.atr_ratio_max)
        & prev_sq.ge(config.min_squeeze_bars)
        & ~frame["squeeze_on"].astype(bool)
        & mom.lt(0)
        & rsi.ge(config.short_rsi_floor)
        & (frame["close_time"] >= start)
        & (frame["close_time"] < end)
    )
    first_eval = max(min(config.warmup_bars - 1, len(frame) - 1), 1)
    candidate.iloc[:first_eval] = False

    # Sequential cooldown over candidates only
    cooldown_sec = config.short_cooldown_h * 3600
    ct_values = frame["close_time"].values
    kept_indices: list[int] = []
    last_short_time = None
    for idx in candidate.values.nonzero()[0]:
        ct = pd.Timestamp(ct_values[idx])
        if last_short_time is not None and (ct - last_short_time).total_seconds() < cooldown_sec:
            continue
        last_short_time = ct
        kept_indices.append(int(idx))

    # Signal construction over the small kept set
    signals: list[Signal] = []
    for idx in kept_indices:
        row = frame.iloc[idx]
        prev = frame.iloc[idx - 1]
        close_time = row["close_time"]
        rsi_val = _safe_value(row, "rsi_14", 50.0)
        mom_val = _safe_value(row, "mom_slope", 0.0)
        atr_val = _safe_value(row, "atr_ratio", 1.0)
        sq_count = int(_safe_value(prev, "squeeze_count", 0.0))
        metadata: dict[str, object] = {
            "strategy": "squeeze_v8.3_short",
            "mom": round(mom_val, 6),
            "rsi": round(rsi_val, 1),
            "atr_ratio": round(atr_val, 2),
            "sq_count": sq_count,
        }
        signal = _build_squeeze_short_signal(
            signal_date=close_time,
            symbol=symbol,
            config=config,
            metadata=metadata,
        )
        signals.append(signal)
    return signals


def _generate_pullback_long_signals_vec(
    frame: pd.DataFrame,
    symbol: str,
    start: datetime,
    end: datetime,
    periods: SqueezeV8Periods,
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
    """Vectorized pullback LONG signal generation for backtesting.

    Produces identical results to ``_generate_pullback_long_signals`` but
    replaces per-row iloc lookback scans with precomputed rolling columns.
    """
    if frame.empty or len(frame) < periods.warmup_bars + 2:
        return []

    # Support: bb_mid if bb_mid > 0 else ema20 (line 866-868)
    bb_mid = frame["_bb_ma_20"].fillna(0.0)
    ema20 = frame["ema_20"]
    support = bb_mid.where(bb_mid > 0, ema20)
    atr = frame["atr_14"]

    # prev_support: _safe_value(prev, "_bb_ma_20", _safe_value(prev, "ema_20", support))
    prev_bb = frame["_bb_ma_20"].shift(1)
    prev_ema = frame["ema_20"].shift(1)
    prev_support = prev_bb.where(prev_bb.notna(), prev_ema.where(prev_ema.notna(), support))

    # Lookback support for rolling: _safe_value(lb_row, "_bb_ma_20", _safe_value(lb_row, "ema_20", 0.0))
    lb_support = frame["_bb_ma_20"].where(
        frame["_bb_ma_20"].notna(),
        frame["ema_20"].where(frame["ema_20"].notna(), 0.0),
    )
    close_above_lb_support = (frame["close"] > lb_support)

    above_window = max(periods.pullback_above_support_bars, 1)
    above_support_window = (
        close_above_lb_support.rolling(above_window, min_periods=1).sum().shift(2)
    )

    impulse_window = max(periods.pullback_impulse_bars - 1, 1)
    impulse_hits = (
        (frame["high"] >= frame["bb_upper"].fillna(float("inf")))
        .rolling(impulse_window, min_periods=1).max().shift(2).fillna(0).astype(bool)
    )

    body = frame["close"] - frame["open"]
    bar_range = frame["high"] - frame["low"]

    candidate = (
        ema20.notna() & atr.notna() & atr.gt(0)
        & frame["ret_72h"].between(10.0, long_ret_max)
        & frame["rsi_14"].fillna(50.0).le(75.0)
        & frame["atr_ratio"].fillna(1.0).le(1.2)
        & frame["mom_slope"].fillna(0.0).gt(0)
        & (frame["close"] > support + 0.3 * atr)
        & body.gt(0) & bar_range.gt(0) & (body / bar_range).ge(0.4)
        & (frame["low"].shift(1) <= prev_support)
        & (frame["close"].shift(1) >= prev_support - 1.0 * atr)
        & above_support_window.ge(above_window / 2)
        & impulse_hits
        & (frame["close_time"] >= start) & (frame["close_time"] < end)
    )
    first_eval = max(periods.warmup_bars, 2)
    candidate.iloc[:first_eval] = False

    # Sequential cooldown over candidates only
    cooldown_sec = long_cooldown_h * 3600
    ct_values = frame["close_time"].values
    kept_indices: list[int] = []
    last_signal_time = None
    for idx in candidate.values.nonzero()[0]:
        ct = pd.Timestamp(ct_values[idx])
        if last_signal_time is not None and (ct - last_signal_time).total_seconds() < cooldown_sec:
            continue
        last_signal_time = ct
        kept_indices.append(int(idx))

    # Signal construction over the small kept set
    signals: list[Signal] = []
    for idx in kept_indices:
        row = frame.iloc[idx]
        close_time = row["close_time"]
        # Recompute metadata values matching _check_pullback_entry
        row_support = float(support.iloc[idx])
        row_atr = float(atr.iloc[idx])
        prev_row = frame.iloc[idx - 1]
        prev_low_val = float(prev_row["low"])
        prev_support_val = float(prev_support.iloc[idx])
        reclaim_strength = (float(row["close"]) - row_support) / row_atr if row_atr > 0 else 0.0
        pullback_depth = (prev_support_val - prev_low_val) / row_atr if row_atr > 0 else 0.0
        metadata: dict[str, object] = {
            "strategy": "pullback_long",
            "ret_72h": round(_safe_value(row, "ret_72h", 0.0), 2),
            "rsi": round(_safe_value(row, "rsi_14", 50.0), 2),
            "atr_ratio": round(_safe_value(row, "atr_ratio", 1.0), 3),
            "pullback_depth": round(pullback_depth, 3),
            "mom_slope": round(_safe_value(row, "mom_slope", 0.0), 6),
            "reclaim_strength": round(reclaim_strength, 3),
        }
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
    state = _SqueezeV8PreviewState(periods=config.periods)
    gate = SourcePeriodGate()
    last_short: datetime | None = None
    warmup_bars = max(config.periods.warmup_bars, 1)

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
            analysis_interval=active_config.analysis_interval,
            min_squeeze_bars=active_config.min_squeeze_bars,
            warmup_bars=active_config.warmup_bars,
        )
        selected_symbols = symbols or list(frames)

        # Squeeze SHORT signals (vectorized backtest path)
        signals: list[Signal] = []
        for symbol in selected_symbols:
            frame = frames.get(symbol)
            if frame is None:
                continue
            signals.extend(
                _generate_squeeze_short_signals_vec(
                    frame, symbol, active_start, active_end, active_config,
                )
            )

    # Pullback LONG signals (vectorized backtest path)
    if active_config.enable_pullback_long:
        available_symbols = [
            s for s in selected_symbols if s in prepared_context.symbols
        ]
        pullback_frames = _build_pullback_feature_frames(
            prepared_context,
            available_symbols,
            analysis_interval=active_config.analysis_interval,
            min_squeeze_bars=active_config.min_squeeze_bars,
            warmup_bars=active_config.warmup_bars,
        ) if available_symbols else {}
        for symbol in selected_symbols:
            pf = pullback_frames.get(symbol)
            if pf is None or pf.empty:
                continue
            signals.extend(
                _generate_pullback_long_signals_vec(
                    pf,
                    symbol,
                    active_start,
                    active_end,
                    active_config.periods,
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

    @property
    def cooldown_hours(self) -> float:
        if self.short_cooldown_h != self.long_cooldown_h:
            raise ValueError(
                f"evaluator cooldown assumes equal short/long cooldown "
                f"(got short={self.short_cooldown_h}, long={self.long_cooldown_h}); "
                f"if they diverge, _enforce_cooldown needs per-side cooldown support"
            )
        return self.short_cooldown_h

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
        if analysis_interval not in _SUPPORTED_ANALYSIS_INTERVALS:
            supported = ", ".join(sorted(_SUPPORTED_ANALYSIS_INTERVALS))
            raise ValueError(f"analysis_interval must be one of: {supported}")
        if analysis_interval != "1h" and sizing_mode == "ridge_v1":
            raise ValueError(
                "ridge_v1 sizing is calibrated only for 1h analysis_interval; use baseline for lower timeframes"
            )
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
        self._periods = SqueezeV8Periods.for_interval(
            analysis_interval,
            min_squeeze_bars=min_squeeze_bars,
            lookback_bars=_BASE_LOOKBACK_BARS,
        )

        self._client: LiveMarketClient | None = None
        self._last_short: dict[str, datetime] = {}
        self._last_long: dict[str, datetime] = {}
        self._source_period_gate = SourcePeriodGate()
        self._states: dict[str, _SqueezeV8PreviewState] = {}
        self._last_committed_hour: dict[str, datetime] = {}
        self._active_symbols: list[str] = list(SYMBOLS)

    @property
    def periods(self) -> SqueezeV8Periods:
        return self._periods

    @property
    def warmup_bars(self) -> int:
        return self._periods.warmup_bars

    @property
    def required_warmup_bars(self) -> int:
        return self.warmup_bars

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        self._warm_up_states()
        print(
            f"SqueezeV8.3 initialized | "
            f"symbols={len(self._active_symbols)} | leverage={self.leverage}x | "
            f"analysis={self.analysis_interval} | poll={self.effective_poll_interval} | "
            f"sizing={self.sizing_mode} | "
            f"SHORT TP/SL=tiered(A:{_TIER_TP['A']}/{_TIER_SL['A']}% "
            f"B:{_TIER_TP['B']}/{_TIER_SL['B']}% C:{_TIER_TP['C']}/{_TIER_SL['C']}%) "
            f"CD={self.short_cooldown_h}h RSI>={self.short_rsi_floor} | "
            f"LONG(pullback) TP/SL={self.long_tp}/{self.long_sl}% "
            f"ret_max={self.long_ret_max}% "
            f"CD={self.long_cooldown_h}h enabled={self.enable_pullback_long} | "
            f"min_sq={self.periods.min_squeeze_bars} bars "
            f"({self.min_squeeze_bars}h-equiv) ATR<={self.atr_ratio_max} | "
            f"max_hold={self.max_holding_hours}h",
            file=sys.stderr,
        )

    def _warm_up_states(self) -> None:
        assert self._client is not None
        now = datetime.now(UTC)
        start = now - timedelta(
            seconds=interval_to_seconds(self.analysis_interval) * (self.periods.lookback_bars + 2)
        )
        failures: list[str] = []
        self._states = {}
        self._last_committed_hour = {}
        active_symbols: list[str] = []
        for symbol in SYMBOLS:
            state = _SqueezeV8PreviewState(periods=self.periods)
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
                if state.candle_count < self.warmup_bars:
                    failures.append(
                        f"{symbol}: only warmed {state.candle_count} candles, need {self.warmup_bars}"
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
                self._states[symbol] = state
                active_symbols.append(symbol)
                print(
                    f"  {symbol}: warmed up {state.candle_count} bars",
                    file=sys.stderr,
                )
            except Exception as exc:
                failures.append(f"{symbol}: warmup fetch failed: {exc}")

        self._active_symbols = active_symbols
        if not self._active_symbols:
            details = "\n".join(f"  - {failure}" for failure in failures)
            raise FatalSignalError(
                "SqueezeV8 warmup failed; no symbols have sufficient Bybit history:\n"
                f"{details}"
            )
        if failures:
            details = "\n".join(f"  - {failure}" for failure in failures)
            print(
                "SqueezeV8 skipped symbols without sufficient Bybit history:\n"
                f"{details}",
                file=sys.stderr,
            )

    def poll(self) -> list[Signal] | None:
        assert self._client is not None
        now = self.current_time()
        signals: list[Signal] = []
        fetch_errors = 0

        with ThreadPoolExecutor(max_workers=6) as pool:
            future_to_symbol = {
                pool.submit(self._check_symbol, symbol, now): symbol
                for symbol in self._active_symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    sigs = future.result()
                    signals.extend(sigs)
                except FatalSignalError:
                    raise
                except Exception as exc:
                    fetch_errors += 1
                    print(f"Error checking {symbol}: {exc}", file=sys.stderr)

        if fetch_errors > 0:
            evaluated = len(SYMBOLS) - fetch_errors
            print(
                f"SqueezeV8: {fetch_errors}/{len(SYMBOLS)} symbols had data fetch errors "
                f"({evaluated} evaluated cleanly)",
                file=sys.stderr,
            )

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
                start=(
                    last_committed
                    or current_hour - timedelta(seconds=interval_to_seconds(self.analysis_interval) * 2)
                ),
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
        if state.candle_count < self.warmup_bars:
            raise FatalSignalError(
                f"{symbol}: only {state.candle_count} candles available, need {self.warmup_bars}"
            )
        if np.isnan(state.last_row.mom_slope) or np.isnan(state.last_row.atr_ratio):
            raise FatalSignalError(
                f"{symbol}: warmup indicators are incomplete after {state.candle_count} candles"
            )

        # Fetch the current open 1h candle snapshot from Bybit
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
            if len(pb_candles) >= self.warmup_bars:
                import pandas as pd
                pb_rows = [{
                    "open_time": c.open_time, "close_time": c.close_time,
                    "open": c.open, "high": c.high, "low": c.low,
                    "close": c.close, "volume": c.volume,
                } for c in pb_candles]
                df_raw = pd.DataFrame(pb_rows).sort_values("open_time").reset_index(drop=True)
                pullback_sigs = self._check_pullback_long(df_raw, symbol, now)
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

        start = now - timedelta(
            seconds=interval_to_seconds(self.analysis_interval) * (self.periods.lookback_bars + 2)
        )
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval=self.analysis_interval,
            start=start,
            end=now,
        )

        if len(candles) < self.warmup_bars:
            raise FatalSignalError(
                f"{symbol}: live fetch returned {len(candles)} candles, need {self.warmup_bars}"
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

        # --- Squeeze SHORT check ---
        df_squeeze = build_squeeze_v8_feature_frame(
            df_raw.copy(),
            analysis_interval=self.analysis_interval,
            min_squeeze_bars=self.min_squeeze_bars,
            warmup_bars=self.warmup_bars,
        )
        last_idx = len(df_squeeze) - 1
        if df_squeeze.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx + 1 < self.warmup_bars:
            raise FatalSignalError(
                f"{symbol}: only {last_idx + 1} closed candles available, need {self.warmup_bars}"
            )

        emitted: list[Signal] = []
        row = df_squeeze.iloc[last_idx]
        prev = df_squeeze.iloc[last_idx - 1]

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
            pullback_sigs = self._check_pullback_long(df_raw, symbol, now)
            emitted.extend(pullback_sigs)

        return emitted

    def _check_pullback_long(
        self,
        df_raw: pd.DataFrame,
        symbol: str,
        now: datetime,
    ) -> list[Signal]:
        """Check pullback LONG conditions on the last closed candle."""
        df_pullback = _build_pullback_feature_frame(
            df_raw.copy(),
            analysis_interval=self.analysis_interval,
            min_squeeze_bars=self.min_squeeze_bars,
            warmup_bars=self.warmup_bars,
        )
        last_idx = len(df_pullback) - 1
        if df_pullback.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx < self.warmup_bars:
            return []

        ok, metadata = _check_pullback_entry(
            df_pullback,
            last_idx,
            self.periods,
            long_ret_max=self.long_ret_max,
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
