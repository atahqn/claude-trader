"""Live signal generator: Breadth Dip-Buy + Selective Momentum LONG.

LONG-only strategy combining two complementary entry patterns within a
breadth-confirmed bull regime (>= 80% of symbols with positive ret_72h):

  1. **Dip-buy**: short-term dip (ret_24h negative) with bullish recovery
     bar after a bearish/neutral previous bar.  TP 4.5% / SL 2%.
  2. **Selective momentum**: very strong, smooth uptrend continuation
     (ret_72h > 20%, RSI 55-65, atr_ratio < 0.85, mom_slope > 0).
     TP 5.0% / SL 2%.  Fires rarely but with high win rate.

The patterns are complementary: dip-buy captures pullback recovery,
selective momentum captures continuation in the strongest trends.
Together they cover more of the bull cycle without quality degradation.

10-symbol universe (non-overlapping with V8.3):
  ETH, SOL, BTC, NEAR, DOT, ADA, APT, RENDER, INJ, LTC

Approximate evaluation (2026-04-02):
  Dev  (58w): pref 42.51, PNL +129%, PF 2.80, MDD 14.3%, 82 trades
  Eval (41w): pref  7.39, PNL +60%,  PF 1.83, MDD 14.6%, 65 trades

Usage:
  python run_strategy_eval.py \\
      --strategy live/breadth_momentum_strategy.py:BreadthMomentumStrategy \\
      --windows eval --approximate
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from marketdata import MarketDataRequest

from backtester.indicators import compute_indicator_frame, required_warmup
from backtester.models import MarketType, PositionType, Signal
from backtester.pipeline import PreparedMarketContext

from .signal_generator import SignalGenerator

SYMBOLS = [
    "ETH/USDT", "SOL/USDT", "BTC/USDT",
    "NEAR/USDT", "DOT/USDT", "ADA/USDT",
    "APT/USDT", "RENDER/USDT", "INJ/USDT", "LTC/USDT",
]

_FEATURE_COLUMNS = (
    "rsi_14", "atr_14", "atr_ratio",
    "ret_72h", "ret_24h",
    "mom_slope", "ema_20",
    "_bb_ma_20", "bb_upper", "bb_lower",
    "body", "body_ratio", "vol_ratio",
)

_WARMUP_BARS = required_warmup(_FEATURE_COLUMNS)


def _safe(row: pd.Series, col: str, default: float) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


def _build_feature_frames(
    prepared_context: PreparedMarketContext,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            raw = prepared_context.for_symbol(symbol).frame
        except KeyError:
            continue
        frames[symbol] = compute_indicator_frame(raw, _FEATURE_COLUMNS)
    return frames


def _compute_breadth_at_time(
    frames: dict[str, pd.DataFrame],
    close_time: datetime,
) -> tuple[int, int]:
    positive = 0
    total = 0
    for frame in frames.values():
        if frame.empty or "close_time" not in frame.columns:
            continue
        mask = frame["close_time"] == close_time
        if not mask.any():
            continue
        row = frame.loc[mask].iloc[0]
        ret_72h = row.get("ret_72h", np.nan)
        if not pd.isna(ret_72h):
            total += 1
            if ret_72h > 0:
                positive += 1
    return positive, total


def _check_dipbuy(row: pd.Series, prev: pd.Series) -> tuple[bool, dict[str, object]]:
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 8.0 or ret_72h > 30.0:
        return False, {}
    ret_24h = _safe(row, "ret_24h", 0.0)
    if ret_24h > 0.5 or ret_24h < -6.0:
        return False, {}
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 65.0 or rsi < 35.0:
        return False, {}
    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 1.2:
        return False, {}
    close = float(row["close"])
    if close <= float(row["ema_20"]):
        return False, {}
    high, low, open_price = float(row["high"]), float(row["low"]), float(row["open"])
    bar_range = high - low
    if bar_range <= 0:
        return False, {}
    body = close - open_price
    if body <= 0 or body / bar_range < 0.3:
        return False, {}
    vol_ratio = _safe(row, "vol_ratio", 1.0)
    if vol_ratio < 0.9:
        return False, {}
    prev_range = float(prev["high"]) - float(prev["low"])
    if prev_range > 0:
        prev_body = float(prev["close"]) - float(prev["open"])
        if prev_body > 0.3 * prev_range:
            return False, {}
    return True, {
        "pattern": "dipbuy",
        "ret_72h": round(ret_72h, 2),
        "ret_24h": round(ret_24h, 2),
        "rsi": round(rsi, 2),
        "atr_ratio": round(atr_ratio, 3),
    }


def _check_selective_momentum(row: pd.Series) -> tuple[bool, dict[str, object]]:
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 20.0 or ret_72h > 40.0:
        return False, {}
    ret_24h = _safe(row, "ret_24h", 0.0)
    if ret_24h < 1.0 or ret_24h > 4.0:
        return False, {}
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 65.0 or rsi < 55.0:
        return False, {}
    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 0.85:
        return False, {}
    close = float(row["close"])
    if close <= float(row["ema_20"]):
        return False, {}
    mom_slope = _safe(row, "mom_slope", 0.0)
    if mom_slope <= 0:
        return False, {}
    high, low, open_price = float(row["high"]), float(row["low"]), float(row["open"])
    bar_range = high - low
    if bar_range <= 0:
        return False, {}
    body = close - open_price
    if body <= 0 or body / bar_range < 0.3:
        return False, {}
    return True, {
        "pattern": "sel_momentum",
        "ret_72h": round(ret_72h, 2),
        "ret_24h": round(ret_24h, 2),
        "rsi": round(rsi, 2),
        "atr_ratio": round(atr_ratio, 3),
        "mom_slope": round(mom_slope, 6),
    }


class BreadthMomentumStrategy(SignalGenerator):
    """Breadth dip-buy + selective momentum LONG strategy."""

    analysis_interval = "1h"

    @property
    def symbols(self) -> list[str]:
        return list(SYMBOLS)

    def __init__(
        self,
        tp_pct: float = 4.5,
        sl_pct: float = 2.0,
        cooldown_h: float = 12.0,
        max_holding_hours: int = 72,
        breadth_min_pct: float = 0.80,
        leverage: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.cooldown_h = cooldown_h
        self.max_holding_hours = max_holding_hours
        self.breadth_min_pct = breadth_min_pct
        self.leverage = leverage

    def __str__(self) -> str:
        return (
            f"BreadthMomentum(breadth>={self.breadth_min_pct:.0%} "
            f"TP{self.tp_pct}/SL{self.sl_pct} "
            f"{len(SYMBOLS)}sym)"
        )

    def market_data_request(self) -> MarketDataRequest:
        return MarketDataRequest.ohlcv_only(interval=self.analysis_interval)

    def poll(self) -> Signal | list[Signal] | None:
        return None

    def generate_backtest_signals(
        self,
        prepared_context: PreparedMarketContext,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[Signal]:
        frames = _build_feature_frames(prepared_context, symbols)
        if not frames:
            return []

        all_times: set[datetime] = set()
        for frame in frames.values():
            if frame.empty or "close_time" not in frame.columns:
                continue
            mask = (frame["close_time"] >= start) & (frame["close_time"] < end)
            all_times.update(frame.loc[mask, "close_time"].tolist())

        breadth_cache: dict[datetime, tuple[int, int]] = {}
        for t in sorted(all_times):
            breadth_cache[t] = _compute_breadth_at_time(frames, t)

        all_signals: list[Signal] = []
        for symbol, frame in frames.items():
            if frame.empty or len(frame) < _WARMUP_BARS + 3:
                continue

            last_signal: datetime | None = None

            for i in range(max(_WARMUP_BARS, 3), len(frame)):
                close_time = frame.iloc[i]["close_time"]
                if close_time < start or close_time >= end:
                    continue

                if last_signal is not None:
                    hours = (close_time - last_signal).total_seconds() / 3600.0
                    if hours < self.cooldown_h:
                        continue

                breadth = breadth_cache.get(close_time)
                if breadth is None:
                    continue
                pos_count, total_count = breadth
                if total_count == 0:
                    continue
                if pos_count / total_count < self.breadth_min_pct:
                    continue

                row = frame.iloc[i]
                prev = frame.iloc[i - 1]

                ok = False
                metadata: dict[str, object] = {}
                ok, metadata = _check_dipbuy(row, prev)
                if not ok:
                    ok, metadata = _check_selective_momentum(row)

                if not ok:
                    continue

                metadata["breadth"] = f"{pos_count}/{total_count}"

                is_momentum = metadata.get("pattern") == "sel_momentum"
                tp = 5.0 if is_momentum else self.tp_pct

                signal = Signal(
                    signal_date=close_time,
                    position_type=PositionType.LONG,
                    ticker=symbol,
                    tp_pct=tp,
                    sl_pct=self.sl_pct,
                    leverage=self.leverage,
                    market_type=MarketType.FUTURES,
                    taker_fee_rate=0.0005,
                    max_holding_hours=self.max_holding_hours,
                    metadata=metadata,
                )
                all_signals.append(signal)
                last_signal = close_time

        all_signals.sort(key=lambda s: s.signal_date)
        return all_signals
