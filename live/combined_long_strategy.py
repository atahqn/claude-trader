"""Combined Breadth + Convergence LONG strategy.

Merges two complementary regime filters into a single LONG strategy
on a shared 10-symbol universe. A signal fires when EITHER regime
gate passes and its corresponding entry pattern matches.

Regime A (breadth): 80%+ of symbols with positive ret_72h
  -> dipbuy (TP 4.5% / SL 2.0%)
  -> selective momentum (TP 5.0% / SL 2.0%)

Regime B (convergence): 70%+ of symbols with positive mom_slope
  -> dipbuy (TP 4.0% / SL 2.0%)
  -> impulse continuation body > 2x ATR (TP 4.0% / SL 2.0%)

Shared 12h per-symbol cooldown prevents overlapping signals.
Breadth entries have priority when both regimes are active.

10-symbol universe (non-overlapping with V8.3):
  ETH, SOL, BTC, NEAR, DOT, ADA, APT, RENDER, INJ, LTC

Approximate evaluation (2026-04-02):
  Dev  (58w): pref 23.15, PNL +177%, PF 2.19, MDD 25.9%, 152 trades
  Eval (41w): pref 10.40, PNL +86%,  PF 1.66, MDD 19.9%, 116 trades
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

_BASE_INDICATORS = (
    "rsi_14", "atr_14", "atr_ratio",
    "ret_72h", "ret_24h",
    "mom_slope", "ema_20",
    "body", "body_ratio", "vol_ratio",
)

_WARMUP_BARS = max(required_warmup(_BASE_INDICATORS), 100)


def _safe(row: pd.Series, col: str, default: float) -> float:
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)


def _compute_cross_asset_state(
    frames: dict[str, pd.DataFrame],
    close_time: datetime,
) -> tuple[int, int, int, int]:
    """Return (pos_mom, total_mom, pos_ret, total_ret)."""
    pos_mom = total_mom = pos_ret = total_ret = 0
    for frame in frames.values():
        if frame.empty or "close_time" not in frame.columns:
            continue
        mask = frame["close_time"] == close_time
        if not mask.any():
            continue
        row = frame.loc[mask].iloc[0]
        ms = row.get("mom_slope", np.nan)
        if not pd.isna(ms):
            total_mom += 1
            if ms > 0:
                pos_mom += 1
        ret = row.get("ret_72h", np.nan)
        if not pd.isna(ret):
            total_ret += 1
            if ret > 0:
                pos_ret += 1
    return pos_mom, total_mom, pos_ret, total_ret


# --- Breadth-regime entries ---

def _breadth_dipbuy(row: pd.Series, prev: pd.Series) -> tuple[bool, dict[str, object], float, float]:
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}, 0, 0
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 8.0 or ret_72h > 30.0:
        return False, {}, 0, 0
    ret_24h = _safe(row, "ret_24h", 0.0)
    if ret_24h > 0.5 or ret_24h < -6.0:
        return False, {}, 0, 0
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 65.0 or rsi < 35.0:
        return False, {}, 0, 0
    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 1.2:
        return False, {}, 0, 0
    close = float(row["close"])
    if close <= float(row["ema_20"]):
        return False, {}, 0, 0
    high, low, open_price = float(row["high"]), float(row["low"]), float(row["open"])
    bar_range = high - low
    if bar_range <= 0:
        return False, {}, 0, 0
    body = close - open_price
    if body <= 0 or body / bar_range < 0.3:
        return False, {}, 0, 0
    vol_ratio = _safe(row, "vol_ratio", 1.0)
    if vol_ratio < 0.9:
        return False, {}, 0, 0
    prev_range = float(prev["high"]) - float(prev["low"])
    if prev_range > 0:
        prev_body = float(prev["close"]) - float(prev["open"])
        if prev_body > 0.3 * prev_range:
            return False, {}, 0, 0
    return True, {"pattern": "breadth_dipbuy", "ret_72h": round(ret_72h, 2), "ret_24h": round(ret_24h, 2)}, 4.5, 2.0


def _breadth_sel_momentum(row: pd.Series) -> tuple[bool, dict[str, object], float, float]:
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}, 0, 0
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 20.0 or ret_72h > 40.0:
        return False, {}, 0, 0
    ret_24h = _safe(row, "ret_24h", 0.0)
    if ret_24h < 1.0 or ret_24h > 4.0:
        return False, {}, 0, 0
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 65.0 or rsi < 55.0:
        return False, {}, 0, 0
    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 0.85:
        return False, {}, 0, 0
    close = float(row["close"])
    if close <= float(row["ema_20"]):
        return False, {}, 0, 0
    mom_slope = _safe(row, "mom_slope", 0.0)
    if mom_slope <= 0:
        return False, {}, 0, 0
    high, low, open_price = float(row["high"]), float(row["low"]), float(row["open"])
    bar_range = high - low
    if bar_range <= 0:
        return False, {}, 0, 0
    body = close - open_price
    if body <= 0 or body / bar_range < 0.3:
        return False, {}, 0, 0
    return True, {"pattern": "breadth_sel_mom", "ret_72h": round(ret_72h, 2)}, 5.0, 2.0


# --- Convergence-regime entries ---

def _conv_dipbuy(row: pd.Series, prev: pd.Series) -> tuple[bool, dict[str, object], float, float]:
    mom_slope = _safe(row, "mom_slope", 0.0)
    if mom_slope <= 0:
        return False, {}, 0, 0
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 5.0 or ret_72h > 35.0:
        return False, {}, 0, 0
    ret_24h = _safe(row, "ret_24h", 0.0)
    if ret_24h < -6.0 or ret_24h > 1.0:
        return False, {}, 0, 0
    prev_range = float(prev["high"]) - float(prev["low"])
    if prev_range > 0:
        prev_body = float(prev["close"]) - float(prev["open"])
        if prev_body > 0.3 * prev_range:
            return False, {}, 0, 0
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 65.0 or rsi < 30.0:
        return False, {}, 0, 0
    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 1.2:
        return False, {}, 0, 0
    close = float(row["close"])
    ema20 = _safe(row, "ema_20", close)
    if close <= ema20:
        return False, {}, 0, 0
    body_ratio = _safe(row, "body_ratio", 0.0)
    if body_ratio < 0.3:
        return False, {}, 0, 0
    vol_ratio = _safe(row, "vol_ratio", 1.0)
    if vol_ratio < 0.9:
        return False, {}, 0, 0
    return True, {"pattern": "conv_dipbuy", "ret_72h": round(ret_72h, 2), "ret_24h": round(ret_24h, 2)}, 4.0, 2.0


def _conv_impulse(row: pd.Series) -> tuple[bool, dict[str, object], float, float]:
    body = _safe(row, "body", 0.0)
    atr = _safe(row, "atr_14", 0.0)
    if atr <= 0 or body <= 0:
        return False, {}, 0, 0
    if body < 2.0 * atr:
        return False, {}, 0, 0
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 3.0 or ret_72h > 30.0:
        return False, {}, 0, 0
    close = float(row["close"])
    ema20 = _safe(row, "ema_20", close)
    if close <= ema20:
        return False, {}, 0, 0
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 70.0 or rsi < 30.0:
        return False, {}, 0, 0
    atr_ratio = _safe(row, "atr_ratio", 1.0)
    if atr_ratio > 1.3:
        return False, {}, 0, 0
    return True, {"pattern": "conv_impulse", "impulse_atr": round(body / atr, 2), "ret_72h": round(ret_72h, 2)}, 4.0, 2.0


# --- Strategy class ---

class CombinedLongStrategy(SignalGenerator):
    """Combined Breadth + Convergence LONG on 10-symbol universe."""

    analysis_interval = "1h"

    @property
    def symbols(self) -> list[str]:
        return list(SYMBOLS)

    def __init__(
        self,
        cooldown_h: float = 12.0,
        max_holding_hours: int = 72,
        convergence_pct: float = 0.70,
        breadth_pct: float = 0.80,
        leverage: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.cooldown_h = cooldown_h
        self.max_holding_hours = max_holding_hours
        self.convergence_pct = convergence_pct
        self.breadth_pct = breadth_pct
        self.leverage = leverage

    def __str__(self) -> str:
        return (
            f"CombinedLong(br>={self.breadth_pct:.0%} "
            f"conv>={self.convergence_pct:.0%} "
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
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                raw = prepared_context.for_symbol(symbol).frame
            except KeyError:
                continue
            if raw.empty:
                continue
            frames[symbol] = compute_indicator_frame(raw, _BASE_INDICATORS)

        if not frames:
            return []

        all_times: set[datetime] = set()
        for frame in frames.values():
            if frame.empty or "close_time" not in frame.columns:
                continue
            mask = (frame["close_time"] >= start) & (frame["close_time"] < end)
            all_times.update(frame.loc[mask, "close_time"].tolist())

        state_cache: dict[datetime, tuple[int, int, int, int]] = {}
        for t in sorted(all_times):
            state_cache[t] = _compute_cross_asset_state(frames, t)

        all_signals: list[Signal] = []

        for symbol, frame in frames.items():
            if frame.empty or len(frame) < _WARMUP_BARS + 3:
                continue

            last_signal: datetime | None = None

            for i in range(max(_WARMUP_BARS, 3), len(frame)):
                row = frame.iloc[i]
                prev = frame.iloc[i - 1]
                close_time = row["close_time"]

                if close_time < start or close_time >= end:
                    continue

                if last_signal is not None:
                    hours = (close_time - last_signal).total_seconds() / 3600.0
                    if hours < self.cooldown_h:
                        continue

                state = state_cache.get(close_time)
                if state is None:
                    continue
                pos_mom, total_mom, pos_ret, total_ret = state

                breadth_on = (total_ret > 0
                              and pos_ret / total_ret >= self.breadth_pct)
                conv_on = (total_mom > 0
                           and pos_mom / total_mom >= self.convergence_pct)

                if not breadth_on and not conv_on:
                    continue

                ok = False
                metadata: dict[str, object] = {}
                tp = sl = 0.0

                if breadth_on:
                    ok, metadata, tp, sl = _breadth_dipbuy(row, prev)
                    if not ok:
                        ok, metadata, tp, sl = _breadth_sel_momentum(row)

                if not ok and conv_on:
                    ok, metadata, tp, sl = _conv_dipbuy(row, prev)
                    if not ok:
                        ok, metadata, tp, sl = _conv_impulse(row)

                if not ok:
                    continue

                if breadth_on:
                    metadata["breadth"] = f"{pos_ret}/{total_ret}"
                if conv_on:
                    metadata["convergence"] = f"{pos_mom}/{total_mom}"

                signal = Signal(
                    signal_date=close_time,
                    position_type=PositionType.LONG,
                    ticker=symbol,
                    tp_pct=tp,
                    sl_pct=sl,
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
