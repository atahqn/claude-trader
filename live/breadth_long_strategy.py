"""Live signal generator: Breadth Dip-Buy LONG Strategy (V21).

LONG-only strategy on non-V8.3 symbols using cross-asset breadth as a
regime filter.  Buys dips within confirmed broad bull markets where at
least 80% of 11 tracked symbols show positive 72-hour returns.

Signal rules:
  - Breadth >= 80% (at least 9/11 symbols with ret_72h > 0)
  - Individual asset: ret_72h in [8%, 30%], ret_24h in [-6%, 0.5%]
  - RSI_14 in [35, 65], ATR ratio <= 1.2
  - Close > EMA20, bullish bar (body_ratio >= 0.3)
  - Previous bar bearish/neutral (dip confirmation)
  - Volume ratio >= 0.9 (conviction filter)
  - TP/SL: 4.0% / 2.0%, cooldown 12h, max hold 72h

11-symbol universe (non-overlapping with V8.3):
  ETH, SOL, BTC, NEAR, WIF, DOT, ADA, APT, RENDER, INJ, LTC

Evaluation (exact mode, 2026-04-01):
  Dev  (58 weeks): pref 15.75, PNL +95.94%, MDD 16.92%, PF 2.07, 92 trades
  Eval (41 weeks): pref  7.94, PNL +62.00%, MDD 15.02%, PF 1.84, 71 trades

Usage:
  python run_strategy_eval.py \\
      --strategy live/breadth_long_strategy.py:BreadthLongStrategy \\
      --windows eval --exact

This is NOT the default live strategy.  ``python -m live.run`` still
launches SqueezeV8Strategy.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
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
    "NEAR/USDT", "WIF/USDT", "DOT/USDT",
    "ADA/USDT", "APT/USDT", "RENDER/USDT",
    "INJ/USDT", "LTC/USDT",
]

_FEATURE_COLUMNS = (
    "rsi_14",
    "atr_14",
    "atr_ratio",
    "ret_72h",
    "ret_24h",
    "mom_slope",
    "ema_20",
    "_bb_ma_20",
    "bb_upper",
    "bb_lower",
    "body",
    "body_ratio",
    "vol_ratio",
)

_WARMUP_BARS = required_warmup(_FEATURE_COLUMNS)


def _safe(row: pd.Series, col: str, default: float) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------

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
    """Return (positive_count, total_count) for ret_72h breadth."""
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


def _check_entry(
    row: pd.Series,
    prev: pd.Series,
) -> tuple[bool, dict[str, object]]:
    """Check individual asset dip-buy entry conditions."""
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

    ema20 = float(row["ema_20"])
    close = float(row["close"])
    if close <= ema20:
        return False, {}

    high = float(row["high"])
    low = float(row["low"])
    open_price = float(row["open"])
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

    mom_slope = _safe(row, "mom_slope", 0.0)
    return True, {
        "strategy": "breadth_dipbuy_v21",
        "ret_72h": round(ret_72h, 2),
        "ret_24h": round(ret_24h, 2),
        "rsi": round(rsi, 2),
        "atr_ratio": round(atr_ratio, 3),
        "mom_slope": round(mom_slope, 6),
        "body_ratio": round(body / bar_range, 3),
        "vol_ratio": round(vol_ratio, 3),
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class BreadthLongStrategy(SignalGenerator):
    """Cross-asset breadth dip-buy LONG strategy (V21)."""

    analysis_interval = "1h"

    @property
    def symbols(self) -> list[str]:
        return list(SYMBOLS)

    def __init__(
        self,
        tp_pct: float = 4.0,
        sl_pct: float = 2.0,
        cooldown_h: float = 12.0,
        max_holding_hours: int = 72,
        breadth_min_pct: float = 0.80,
        **kwargs: Any,
    ) -> None:
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.cooldown_h = cooldown_h
        self.max_holding_hours = max_holding_hours
        self.breadth_min_pct = breadth_min_pct

    def __str__(self) -> str:
        return (
            f"BreadthLong(breadth>={self.breadth_min_pct:.0%} "
            f"TP{self.tp_pct}/SL{self.sl_pct} "
            f"CD{self.cooldown_h}h)"
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

        # Pre-compute breadth for all timestamps in the window
        all_times: set[datetime] = set()
        for frame in frames.values():
            if frame.empty or "close_time" not in frame.columns:
                continue
            mask = (frame["close_time"] >= start) & (frame["close_time"] < end)
            all_times.update(frame.loc[mask, "close_time"].tolist())

        breadth_cache: dict[datetime, tuple[int, int]] = {}
        for t in sorted(all_times):
            breadth_cache[t] = _compute_breadth_at_time(frames, t)

        # Generate signals per symbol
        all_signals: list[Signal] = []
        for symbol, frame in frames.items():
            if frame.empty or len(frame) < _WARMUP_BARS + 3:
                continue

            last_signal: datetime | None = None

            for i in range(max(_WARMUP_BARS, 3), len(frame)):
                close_time = frame.iloc[i]["close_time"]
                if close_time < start or close_time >= end:
                    continue

                # Cooldown
                if last_signal is not None:
                    hours = (close_time - last_signal).total_seconds() / 3600.0
                    if hours < self.cooldown_h:
                        continue

                # Breadth check
                breadth = breadth_cache.get(close_time)
                if breadth is None:
                    continue
                pos_count, total_count = breadth
                if total_count == 0:
                    continue
                breadth_pct = pos_count / total_count
                if breadth_pct < self.breadth_min_pct:
                    continue

                # Individual entry
                row = frame.iloc[i]
                prev = frame.iloc[i - 1]
                ok, metadata = _check_entry(row, prev)
                if not ok:
                    continue

                metadata["breadth"] = f"{pos_count}/{total_count}"

                signal = Signal(
                    signal_date=close_time,
                    position_type=PositionType.LONG,
                    ticker=symbol,
                    tp_pct=self.tp_pct,
                    sl_pct=self.sl_pct,
                    leverage=1.0,
                    market_type=MarketType.FUTURES,
                    taker_fee_rate=0.0005,
                    max_holding_hours=self.max_holding_hours,
                    metadata=metadata,
                )
                all_signals.append(signal)
                last_signal = close_time

        all_signals.sort(key=lambda s: s.signal_date)
        return all_signals
