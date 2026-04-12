"""Combined Breadth + Convergence + BTC Structure LONG strategy.

Merges three complementary regime filters into a single LONG strategy
on a shared 10-symbol universe.

Regime A (breadth): 80%+ of symbols with positive ret_72h
  -> dipbuy (TP 4.5% / SL 2.0%)
  -> selective momentum (TP 5.0% / SL 2.0%)

Regime B (convergence): 70%+ of symbols with positive mom_slope
  -> dipbuy (TP 4.0% / SL 2.0%)  [gated by BTC daily structure]
  -> impulse continuation body > 2x ATR (TP 4.0% / SL 2.0%)
     [gated by BTC daily structure]

Regime C (BTC structure): BTC global_continuation_long_flag OR
  major_global_bullish_confluence_flag, combined with cross-asset
  convergence (70%+) or breadth (80%+)
  -> struct_dipbuy (TP 4.0% / SL 2.0%)  [convergence path]
  -> struct_impulse body > 2x ATR (TP 4.0% / SL 2.0%)
     [convergence or breadth path]

Regime C fires at different times than A/B — when BTC's multi-year
swing structure is bullish but breadth/convergence alone may not be
active. Adds ~14 high-quality trades on eval (+26% PNL) without
increasing MDD.

BTC structure gate (conv_no_bearish): convergence entries (Regime B)
are blocked when BTC daily market_bias_after_close == "bearish".

Shared 12h per-symbol cooldown prevents overlapping signals.
Breadth entries have priority, then convergence, then BTC structure.

10-symbol universe (non-overlapping with V8.3):
  ETH, SOL, BTC, NEAR, DOT, ADA, APT, RENDER, INJ, LTC

Exact evaluation (2026-04-03, merged + heuristic_v1):
  Eval Pref 35.50 (PNL +115.02%, MDD 11.9%, PF 2.02)
  vs prior CombinedLong-only Eval Pref 22.95

Supports sizing_mode:
  "baseline"       — all signals get size_multiplier=1.0
  "heuristic_v1"   — pattern-based sizing with dip-depth and cluster
                     adjustments (deployed default, applies to Regime
                     A/B signals only; Regime C signals use 1.0)

Supports tp_sl_mode:
  "baseline"       — fixed per-pattern TP/SL (deployed default)
  "tiered_v1"      — quality-tiered TP/SL (failed on eval, do not use)
"""

from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .auth_client import LiveMarketClient

import numpy as np
import pandas as pd
from marketdata import MarketDataRequest

from backtester.indicators import compute_indicator_frame, required_warmup
from backtester.models import MarketType, PositionType, Signal
from backtester.pipeline import PreparedMarketContext
from btc_structure import DailyStructureProvider

from .signal_generator import FatalSignalError, SignalGenerator

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
    "volume_delta",
)

_WARMUP_BARS = max(required_warmup(_BASE_INDICATORS), 100)

_BTC_STRUCTURE_COLUMNS = [
    "market_bias_after_close",
    "global_continuation_long_flag",
    "major_global_bullish_confluence_flag",
]


def _btc_structure_is_bearish(row: pd.Series) -> bool:
    """Return True if BTC daily structure regime is bearish."""
    bias = row.get("market_bias_after_close")
    return isinstance(bias, str) and bias == "bearish"


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


def _build_cross_asset_cache(
    frames: dict[str, pd.DataFrame],
    timestamps: list[datetime],
) -> dict[datetime, tuple[int, int, int, int]]:
    """Batch-compute cross-asset state using time-indexed lookups."""
    mom_index: dict[str, dict[datetime, float]] = {}
    ret_index: dict[str, dict[datetime, float]] = {}
    for symbol, frame in frames.items():
        if frame.empty or "close_time" not in frame.columns:
            continue
        ct = frame["close_time"]
        if "mom_slope" in frame.columns:
            mom_index[symbol] = dict(zip(ct, frame["mom_slope"]))
        if "ret_72h" in frame.columns:
            ret_index[symbol] = dict(zip(ct, frame["ret_72h"]))
    cache: dict[datetime, tuple[int, int, int, int]] = {}
    for t in timestamps:
        pos_mom = total_mom = pos_ret = total_ret = 0
        for sym_mom in mom_index.values():
            v = sym_mom.get(t)
            if v is not None and not pd.isna(v):
                total_mom += 1
                if v > 0:
                    pos_mom += 1
        for sym_ret in ret_index.values():
            v = sym_ret.get(t)
            if v is not None and not pd.isna(v):
                total_ret += 1
                if v > 0:
                    pos_ret += 1
        cache[t] = (pos_mom, total_mom, pos_ret, total_ret)
    return cache


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
    vd = _safe(row, "volume_delta", 0.0)
    if vd <= 0:
        return False, {}, 0, 0
    return True, {"pattern": "conv_dipbuy", "ret_72h": round(ret_72h, 2), "ret_24h": round(ret_24h, 2)}, 4.0, 2.0


def _struct_dipbuy(row: pd.Series, prev: pd.Series) -> tuple[bool, dict[str, object], float, float]:
    """Dip-buy gated by BTC structure (continuation/confluence)."""
    if pd.isna(row.get("ema_20")) or pd.isna(row.get("atr_14")):
        return False, {}, 0, 0
    ret_72h = _safe(row, "ret_72h", 0.0)
    if ret_72h < 5.0 or ret_72h > 35.0:
        return False, {}, 0, 0
    ret_24h = _safe(row, "ret_24h", 0.0)
    if ret_24h > 0.5 or ret_24h < -6.0:
        return False, {}, 0, 0
    rsi = _safe(row, "rsi_14", 50.0)
    if rsi > 65.0 or rsi < 30.0:
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
    return True, {"pattern": "struct_dipbuy", "ret_72h": round(ret_72h, 2), "ret_24h": round(ret_24h, 2)}, 4.0, 2.0


def _struct_impulse(row: pd.Series) -> tuple[bool, dict[str, object], float, float]:
    """Impulse continuation gated by BTC structure."""
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
    mom_slope = _safe(row, "mom_slope", 0.0)
    if mom_slope <= 0:
        return False, {}, 0, 0
    vd = _safe(row, "volume_delta", 0.0)
    if vd <= 0:
        return False, {}, 0, 0
    return True, {"pattern": "struct_impulse", "impulse_atr": round(body / atr, 2), "ret_72h": round(ret_72h, 2)}, 4.0, 2.0


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
    vd = _safe(row, "volume_delta", 0.0)
    if vd <= 0:
        return False, {}, 0, 0
    return True, {"pattern": "conv_impulse", "impulse_atr": round(body / atr, 2), "ret_72h": round(ret_72h, 2)}, 4.0, 2.0


# --- Heuristic V1 sizing ---
#
# Pattern-based multiplier from cross-split trade analysis:
#   conv_impulse  PF ~2.15 (dev+eval) -> upweight
#   breadth_sel_mom PF ~3.0 (small N) -> moderate upweight
#   breadth_dipbuy PF ~2.1  (standard) -> neutral
#   conv_dipbuy   PF ~1.25 (weakest)  -> downweight
#
# Dip-depth adjustment: deeper ret_24h -> better outcomes on both splits.
# Cluster dilution: many simultaneous signals -> reduce each.

_PATTERN_BASE_MULT: dict[str, float] = {
    "conv_impulse": 1.30,
    "breadth_sel_mom": 1.15,
    "breadth_dipbuy": 1.00,
    "conv_dipbuy": 0.75,
}

_SIZE_CLIP_LO = 0.50
_SIZE_CLIP_HI = 1.60


def _heuristic_v1_size(
    pattern: str,
    ret_24h: float | None,
    cluster_count: int,
) -> float:
    base = _PATTERN_BASE_MULT.get(pattern, 1.0)

    # Dip-depth adjustment (dipbuy patterns only)
    if "dipbuy" in pattern and ret_24h is not None:
        if ret_24h <= -3.0:
            base *= 1.10
        elif ret_24h <= -1.0:
            pass  # neutral
        else:
            base *= 0.90

    # Cluster dilution: log-scale penalty for simultaneous signals
    if cluster_count >= 4:
        base *= 0.80
    elif cluster_count >= 3:
        base *= 0.90

    return max(_SIZE_CLIP_LO, min(_SIZE_CLIP_HI, base))


def _apply_heuristic_v1_sizing(signals: list[Signal]) -> list[Signal]:
    time_counts: Counter[datetime] = Counter(s.signal_date for s in signals)
    result: list[Signal] = []
    for sig in signals:
        meta = dict(sig.metadata)
        pattern = meta.get("pattern", "")
        ret_24h = meta.get("ret_24h")
        cluster = time_counts[sig.signal_date]
        mult = _heuristic_v1_size(pattern, ret_24h, cluster)
        meta["size_model"] = "heuristic_v1"
        meta["size_mult"] = round(mult, 4)
        meta["cluster_count"] = cluster
        result.append(replace(sig, size_multiplier=mult, metadata=meta))
    return result


# --- Exposure cap ---

_MAX_TOTAL_SIZE_PER_TIMESTAMP = 2.5


def _apply_exposure_cap(signals: list[Signal]) -> list[Signal]:
    """Scale down signals when total size_multiplier exceeds the cap."""
    from itertools import groupby
    result: list[Signal] = []
    for _, group in groupby(signals, key=lambda s: s.signal_date):
        batch = list(group)
        total = sum(s.size_multiplier for s in batch)
        if total > _MAX_TOTAL_SIZE_PER_TIMESTAMP:
            scale = _MAX_TOTAL_SIZE_PER_TIMESTAMP / total
            for sig in batch:
                new_mult = sig.size_multiplier * scale
                meta = dict(sig.metadata)
                meta["size_mult"] = round(new_mult, 4)
                result.append(replace(sig, size_multiplier=new_mult, metadata=meta))
        else:
            result.extend(batch)
    return result


# --- Tiered V1 TP/SL ---
#
# Quality-tiered exits based on pattern quality:
#   conv_dipbuy (weakest PF ~1.25): tighter SL 1.5% to cut losses faster
#   conv_impulse with impulse_atr >= 2.5: wider TP 5.0% to let winners run
#   Everything else: unchanged from baseline

def _apply_tiered_v1_tp_sl(signals: list[Signal]) -> list[Signal]:
    result: list[Signal] = []
    for sig in signals:
        meta = dict(sig.metadata)
        pattern = meta.get("pattern", "")
        tp = sig.tp_pct
        sl = sig.sl_pct

        if pattern == "conv_dipbuy":
            sl = 1.5
            meta["tp_sl_tier"] = "tight_sl"
        elif pattern == "conv_impulse":
            impulse_atr = meta.get("impulse_atr", 0.0)
            if impulse_atr >= 2.5:
                tp = 5.0
                meta["tp_sl_tier"] = "wide_tp"
            else:
                meta["tp_sl_tier"] = "standard"
        else:
            meta["tp_sl_tier"] = "standard"

        result.append(replace(sig, tp_pct=tp, sl_pct=sl, metadata=meta))
    return result


_VALID_SIZING_MODES = ("baseline", "heuristic_v1")
_VALID_TP_SL_MODES = ("baseline", "tiered_v1")


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
        sizing_mode: str = "heuristic_v1",
        tp_sl_mode: str = "baseline",
        **kwargs: Any,
    ) -> None:
        if sizing_mode not in _VALID_SIZING_MODES:
            raise ValueError(f"sizing_mode must be one of {_VALID_SIZING_MODES}")
        if tp_sl_mode not in _VALID_TP_SL_MODES:
            raise ValueError(f"tp_sl_mode must be one of {_VALID_TP_SL_MODES}")
        self.cooldown_h = cooldown_h
        self.max_holding_hours = max_holding_hours
        self.convergence_pct = convergence_pct
        self.breadth_pct = breadth_pct
        self.leverage = leverage
        self.sizing_mode = sizing_mode
        self.tp_sl_mode = tp_sl_mode
        self._structure = DailyStructureProvider(columns=_BTC_STRUCTURE_COLUMNS)
        self._client: LiveMarketClient | None = None
        self._last_signal: dict[str, datetime] = {}
        self._candle_buffers: dict[str, list] = {}
        self._last_poll_time: datetime | None = None

    @property
    def cooldown_hours(self) -> float:
        return self.cooldown_h

    def __str__(self) -> str:
        parts = [
            f"CombinedLong(br>={self.breadth_pct:.0%}",
            f"conv>={self.convergence_pct:.0%}",
            "btc_struct+gate",
            f"{len(SYMBOLS)}sym",
        ]
        if self.sizing_mode != "baseline":
            parts.append(f"size={self.sizing_mode}")
        if self.tp_sl_mode != "baseline":
            parts.append(f"tpsl={self.tp_sl_mode}")
        return " ".join(parts) + ")"

    def market_data_request(self) -> MarketDataRequest:
        return MarketDataRequest.ohlcv_only(interval=self.analysis_interval)

    # ------------------------------------------------------------------
    # Live trading: setup / warmup / poll
    # ------------------------------------------------------------------

    def setup(self, client: "LiveMarketClient") -> None:
        self._client = client
        self._structure = DailyStructureProvider(
            client, columns=_BTC_STRUCTURE_COLUMNS,
        )
        self._last_signal = {}
        self._warm_up()
        print(
            f"CombinedLong initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"analysis={self.analysis_interval} | "
            f"sizing={self.sizing_mode} | "
            f"breadth>={self.breadth_pct:.0%} | conv>={self.convergence_pct:.0%} | "
            f"cooldown={self.cooldown_h}h | "
            f"max_hold={self.max_holding_hours}h",
            file=sys.stderr,
        )

    def _warm_up(self) -> None:
        """Validate sufficient history and indicator quality for every symbol."""
        assert self._client is not None
        now = datetime.now(UTC)
        start = now - timedelta(hours=_WARMUP_BARS + 20)
        failures: list[str] = []

        for symbol in SYMBOLS:
            try:
                candles = self._client.fetch_klines(
                    symbol=symbol.replace("/", ""),
                    interval=self.analysis_interval,
                    start=start,
                    end=now,
                )
                if not candles:
                    failures.append(f"{symbol}: no candle data returned")
                    continue
                self._candle_buffers[symbol] = candles
                frame = self._candles_to_indicator_frame(candles)
                if frame is None:
                    failures.append(f"{symbol}: no candle data returned")
                    continue
                last_idx = len(frame) - 1
                if frame.iloc[last_idx]["close_time"] > now:
                    last_idx -= 1
                if last_idx < _WARMUP_BARS:
                    failures.append(
                        f"{symbol}: only {last_idx + 1} closed candles, "
                        f"need {_WARMUP_BARS}"
                    )
                    continue
                last_row = frame.iloc[last_idx]
                if (
                    pd.isna(last_row.get("mom_slope"))
                    or pd.isna(last_row.get("atr_ratio"))
                ):
                    failures.append(
                        f"{symbol}: warmup indicators incomplete after "
                        f"{last_idx + 1} candles"
                    )
                    continue
                print(
                    f"  {symbol}: warmed up {last_idx + 1} bars",
                    file=sys.stderr,
                )
            except Exception as exc:
                failures.append(f"{symbol}: warmup fetch failed: {exc}")

        self._last_poll_time = now

        # Ensure BTC structure is ready before first poll
        self._structure.ensure_computed_until(now)

        if failures:
            details = "\n".join(f"  - {f}" for f in failures)
            raise FatalSignalError(
                "CombinedLong warmup failed; refusing to trade without full "
                "candle history:\n" + details
            )

    @staticmethod
    def _candles_to_indicator_frame(candles: list) -> pd.DataFrame | None:
        """Convert raw candle list to indicator DataFrame."""
        if not candles:
            return None
        rows = [
            {
                "open_time": c.open_time,
                "close_time": c.close_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "taker_buy_volume": c.taker_buy_volume,
            }
            for c in candles
        ]
        df = pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)
        if df.empty:
            return None
        return compute_indicator_frame(df, _BASE_INDICATORS)

    def _fetch_symbol_frame(
        self, symbol: str, now: datetime,
    ) -> pd.DataFrame | None:
        """Fetch new candles incrementally, update buffer, compute indicators."""
        assert self._client is not None
        buf = self._candle_buffers.get(symbol)
        if buf and self._last_poll_time is not None:
            # Incremental: only fetch candles since last poll (with 2h overlap
            # buffer for safety against gaps).
            fetch_start = self._last_poll_time - timedelta(hours=2)
            new_candles = self._client.fetch_klines(
                symbol=symbol.replace("/", ""),
                interval=self.analysis_interval,
                start=fetch_start,
                end=now,
            )
            if new_candles:
                existing_times = {c.open_time for c in buf}
                for c in new_candles:
                    if c.open_time not in existing_times:
                        buf.append(c)
                        existing_times.add(c.open_time)
                # Trim to prevent unbounded growth
                max_buf = _WARMUP_BARS + 30
                if len(buf) > max_buf:
                    buf[:] = buf[-max_buf:]
        else:
            # First fetch (no buffer yet)
            start = now - timedelta(hours=_WARMUP_BARS + 20)
            buf = self._client.fetch_klines(
                symbol=symbol.replace("/", ""),
                interval=self.analysis_interval,
                start=start,
                end=now,
            )
            if not buf:
                return None
            self._candle_buffers[symbol] = buf

        return self._candles_to_indicator_frame(buf)

    def poll(self) -> Signal | list[Signal] | None:
        assert self._client is not None
        now = self.current_time()

        # -- 1. Fetch indicator frames for all symbols in parallel -----------
        frames: dict[str, pd.DataFrame] = {}
        fetch_errors = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            future_to_symbol = {
                pool.submit(self._fetch_symbol_frame, symbol, now): symbol
                for symbol in SYMBOLS
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    frame = future.result()
                    if frame is not None:
                        frames[symbol] = frame
                except FatalSignalError:
                    raise
                except Exception as exc:
                    fetch_errors += 1
                    print(
                        f"CombinedLong: error fetching {symbol}: {exc}",
                        file=sys.stderr,
                    )

        if fetch_errors > 0:
            evaluated = len(SYMBOLS) - fetch_errors
            print(
                f"CombinedLong: {fetch_errors}/{len(SYMBOLS)} symbols had data fetch errors "
                f"({evaluated} evaluated cleanly)",
                file=sys.stderr,
            )

        self._last_poll_time = now

        if not frames:
            return None

        # -- 2. Merge BTC structure onto all frames --------------------------
        self._structure.refresh_if_stale(now)
        btc_frame = frames.get("BTC/USDT")
        if btc_frame is None:
            btc_frame = next(iter(frames.values()))
        data_cutoff = pd.Timestamp(
            btc_frame["close_time"].max(),
        ).to_pydatetime()
        self._structure.ensure_computed_until(data_cutoff)

        for symbol in list(frames):
            frames[symbol] = self._structure.merge_onto(
                frames[symbol], _BTC_STRUCTURE_COLUMNS, cutoff=data_cutoff,
            )

        # -- 3. Determine last closed-bar index per symbol -------------------
        last_indices: dict[str, int] = {}
        latest_ct: datetime | None = None
        for symbol, frame in frames.items():
            last_idx = len(frame) - 1
            if frame.iloc[last_idx]["close_time"] > now:
                last_idx -= 1
            if last_idx < 0:
                continue
            last_indices[symbol] = last_idx
            ct = frame.iloc[last_idx]["close_time"]
            if latest_ct is None or ct > latest_ct:
                latest_ct = ct

        if latest_ct is None:
            return None

        # -- 4. Cross-asset state at latest closed bar -----------------------
        pos_mom, total_mom, pos_ret, total_ret = _compute_cross_asset_state(
            frames, latest_ct,
        )

        breadth_on = (
            total_ret > 0 and pos_ret / total_ret >= self.breadth_pct
        )
        conv_on = (
            total_mom > 0 and pos_mom / total_mom >= self.convergence_pct
        )

        # -- 5. Check each symbol for entry signals -------------------------
        signals: list[Signal] = []
        for symbol, frame in frames.items():
            last_idx = last_indices.get(symbol)
            if last_idx is None or last_idx < max(_WARMUP_BARS, 3):
                continue

            row = frame.iloc[last_idx]
            prev = frame.iloc[last_idx - 1]
            close_time = row["close_time"]

            # Cooldown
            last = self._last_signal.get(symbol)
            if last is not None:
                hours = (close_time - last).total_seconds() / 3600.0
                if hours < self.cooldown_h:
                    continue

            # BTC structure flags
            btc_cont = bool(row.get("global_continuation_long_flag", False))
            btc_conf = bool(
                row.get("major_global_bullish_confluence_flag", False),
            )
            btc_struct_on = btc_cont or btc_conf

            if not breadth_on and not conv_on and not btc_struct_on:
                continue

            ok = False
            metadata: dict[str, object] = {}
            tp = sl = 0.0
            is_convergence = False

            # Priority 1: breadth entries
            if breadth_on:
                ok, metadata, tp, sl = _breadth_dipbuy(row, prev)
                if not ok:
                    ok, metadata, tp, sl = _breadth_sel_momentum(row)

            # Priority 2: convergence entries
            if not ok and conv_on:
                ok, metadata, tp, sl = _conv_dipbuy(row, prev)
                if ok:
                    is_convergence = True
                if not ok:
                    ok, metadata, tp, sl = _conv_impulse(row)
                    if ok:
                        is_convergence = True

            # Priority 3: BTC structure entries
            if not ok and btc_struct_on:
                if conv_on:
                    ok, metadata, tp, sl = _struct_dipbuy(row, prev)
                    if not ok:
                        ok, metadata, tp, sl = _struct_impulse(row)
                elif breadth_on:
                    ok, metadata, tp, sl = _struct_impulse(row)

            if not ok:
                continue

            # BTC structure gate: block convergence entries during bearish
            if is_convergence and _btc_structure_is_bearish(row):
                continue

            if breadth_on:
                metadata["breadth"] = f"{pos_ret}/{total_ret}"
            if conv_on:
                metadata["convergence"] = f"{pos_mom}/{total_mom}"
            if btc_struct_on:
                metadata["btc_cont"] = btc_cont
                metadata["btc_conf"] = btc_conf
            metadata["btc_bias"] = row.get("market_bias_after_close")

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
            signals.append(signal)
            self._last_signal[symbol] = close_time

        if not signals:
            return None

        # -- 6. Post-processing: TP/SL mode then sizing ---------------------
        signals.sort(key=lambda s: s.signal_date)
        if self.tp_sl_mode == "tiered_v1":
            signals = _apply_tiered_v1_tp_sl(signals)
        if self.sizing_mode == "heuristic_v1":
            signals = _apply_heuristic_v1_sizing(signals)

        # -- 7. Exposure cap: limit total size per timestamp ----------------
        signals = _apply_exposure_cap(signals)

        return signals if signals else None

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

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

        # Compute BTC structure and merge onto all frames
        btc_frame = frames.get("BTC/USDT")
        if btc_frame is None:
            btc_frame = next(iter(frames.values()))
        data_cutoff = pd.Timestamp(btc_frame["close_time"].max()).to_pydatetime()
        self._structure.ensure_computed_until(data_cutoff)

        for symbol, frame in list(frames.items()):
            frames[symbol] = self._structure.merge_onto(
                frame,
                _BTC_STRUCTURE_COLUMNS,
                cutoff=data_cutoff,
            )

        all_times: set[datetime] = set()
        for frame in frames.values():
            if frame.empty or "close_time" not in frame.columns:
                continue
            mask = (frame["close_time"] >= start) & (frame["close_time"] < end)
            all_times.update(frame.loc[mask, "close_time"].tolist())

        state_cache = _build_cross_asset_cache(frames, sorted(all_times))

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

                # BTC structure flags
                btc_cont = bool(row.get("global_continuation_long_flag", False))
                btc_conf = bool(row.get("major_global_bullish_confluence_flag", False))
                btc_struct_on = btc_cont or btc_conf

                if not breadth_on and not conv_on and not btc_struct_on:
                    continue

                ok = False
                metadata: dict[str, object] = {}
                tp = sl = 0.0
                is_convergence = False

                # Priority 1: breadth entries
                if breadth_on:
                    ok, metadata, tp, sl = _breadth_dipbuy(row, prev)
                    if not ok:
                        ok, metadata, tp, sl = _breadth_sel_momentum(row)

                # Priority 2: convergence entries
                if not ok and conv_on:
                    ok, metadata, tp, sl = _conv_dipbuy(row, prev)
                    if ok:
                        is_convergence = True
                    if not ok:
                        ok, metadata, tp, sl = _conv_impulse(row)
                        if ok:
                            is_convergence = True

                # Priority 3: BTC structure entries (fires when breadth/conv miss)
                if not ok and btc_struct_on:
                    if conv_on:
                        ok, metadata, tp, sl = _struct_dipbuy(row, prev)
                        if not ok:
                            ok, metadata, tp, sl = _struct_impulse(row)
                    elif breadth_on:
                        ok, metadata, tp, sl = _struct_impulse(row)

                if not ok:
                    continue

                # BTC structure gate: block convergence entries during bearish
                if is_convergence and _btc_structure_is_bearish(row):
                    continue

                if breadth_on:
                    metadata["breadth"] = f"{pos_ret}/{total_ret}"
                if conv_on:
                    metadata["convergence"] = f"{pos_mom}/{total_mom}"
                if btc_struct_on:
                    metadata["btc_cont"] = btc_cont
                    metadata["btc_conf"] = btc_conf
                metadata["btc_bias"] = row.get("market_bias_after_close")

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

        # Post-processing: dynamic TP/SL then sizing
        if self.tp_sl_mode == "tiered_v1":
            all_signals = _apply_tiered_v1_tp_sl(all_signals)
        if self.sizing_mode == "heuristic_v1":
            all_signals = _apply_heuristic_v1_sizing(all_signals)

        # Exposure cap: limit total size per timestamp
        all_signals = _apply_exposure_cap(all_signals)

        return all_signals
