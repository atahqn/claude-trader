from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import BtcStructureConfig


# ---------------------------------------------------------------------------
# Bar data structures — replace pd.Series row access in the hot loop
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BarData:
    index: int = 0
    close_time: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    atr: float = 0.0
    rolling_highs: dict[int, float] = field(default_factory=dict)
    rolling_lows: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class BarArrays:
    close_time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    atr: np.ndarray
    rolling_highs: dict[int, np.ndarray]
    rolling_lows: dict[int, np.ndarray]
    windows: tuple[int, ...]
    length: int

    def bar_at(self, i: int) -> BarData:
        """Create a fresh BarData — used for staleness re-anchoring."""
        return BarData(
            index=i,
            close_time=self.close_time[i],
            open=self.open[i],
            high=self.high[i],
            low=self.low[i],
            close=self.close[i],
            volume=self.volume[i],
            atr=self.atr[i],
            rolling_highs={w: self.rolling_highs[w][i] for w in self.windows},
            rolling_lows={w: self.rolling_lows[w][i] for w in self.windows},
        )

    def fill_bar(self, i: int, bar: BarData) -> None:
        """Update a mutable BarData in place — avoids allocation in hot loop."""
        bar.index = i
        bar.close_time = self.close_time[i]
        bar.open = self.open[i]
        bar.high = self.high[i]
        bar.low = self.low[i]
        bar.close = self.close[i]
        bar.volume = self.volume[i]
        bar.atr = self.atr[i]
        rh = bar.rolling_highs
        rl = bar.rolling_lows
        for w in self.windows:
            rh[w] = self.rolling_highs[w][i]
            rl[w] = self.rolling_lows[w][i]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StructureArtifacts:
    ohlcv: pd.DataFrame
    features: pd.DataFrame
    candidate_highs: pd.DataFrame
    candidate_lows: pd.DataFrame
    confirmed_highs: pd.DataFrame
    confirmed_lows: pd.DataFrame
    break_attempt_highs: pd.DataFrame
    break_attempt_lows: pd.DataFrame
    structure_breaks: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class StructureCheckpoint:
    """Snapshot of engine state after processing bar at ``resume_from - 1``.

    Allows ``simulate_btc_structure`` to resume from bar ``resume_from``
    without re-processing earlier bars, while producing bit-identical results.
    """

    resume_from: int  # next bar index to process
    state: dict[str, Any]  # 7-key engine state dict
    stats: dict[str, Any]  # 11-key stats dict (event lists + counters)
    feature_rows: list[dict[str, Any]]  # accumulated feature row dicts [0..resume_from)
    prefix_hash: str  # hex digest over processed OHLCV prefix
    config_fingerprint: str  # repr(config) — invalidates on config change


def _ohlcv_prefix_hash(ohlcv: pd.DataFrame, n: int) -> str:
    """Hash the first *n* rows' OHLCV+volume values for checkpoint identity.

    Uses ``to_records`` to preserve per-column dtype (int64 timestamps,
    float64 prices/volume) instead of ``.values`` which upcasts the
    mixed-dtype frame to a single float64 array, losing nanosecond
    timestamp precision.
    """
    cols = ["close_time", "open", "high", "low", "close", "volume"]
    subset = ohlcv.iloc[:n][cols].copy()
    subset["close_time"] = subset["close_time"].astype("int64")
    return hashlib.sha256(subset.to_records(index=False).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Vectorised pre-computation (ATR, rolling levels)
# ---------------------------------------------------------------------------

def causal_atr(frame: pd.DataFrame, window: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=1).mean()


def _build_rolling_levels(frame: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = frame.copy()
    for window in windows:
        min_periods = max(1, min(int(window), max(5, int(window) // 4)))
        out[f"rolling_high_{window}"] = (
            out["high"].shift(1).rolling(window=window, min_periods=min_periods).max()
        )
        out[f"rolling_low_{window}"] = (
            out["low"].shift(1).rolling(window=window, min_periods=min_periods).min()
        )
    return out


def _extract_arrays(work: pd.DataFrame, config: BtcStructureConfig) -> BarArrays:
    windows = config.level_windows
    # Preserve UTC: convert to Python datetime list so pd.Timestamp()
    # round-trips as tz-aware throughout the simulation loop.
    ct_series = pd.to_datetime(work["close_time"], utc=True)
    return BarArrays(
        close_time=np.array(pd.DatetimeIndex(ct_series).to_pydatetime()),
        open=work["open"].values.astype(np.float64),
        high=work["high"].values.astype(np.float64),
        low=work["low"].values.astype(np.float64),
        close=work["close"].values.astype(np.float64),
        volume=work["volume"].values.astype(np.float64),
        atr=work["atr"].values.astype(np.float64),
        rolling_highs={w: work[f"rolling_high_{w}"].values.astype(np.float64) for w in windows},
        rolling_lows={w: work[f"rolling_low_{w}"].values.astype(np.float64) for w in windows},
        windows=windows,
        length=len(work),
    )


# ---------------------------------------------------------------------------
# Structure labelling helpers
# ---------------------------------------------------------------------------

def _label_tolerance(reference_value: float, atr_value: float, config: BtcStructureConfig) -> float:
    atr_part = 0.0 if math.isnan(atr_value) else atr_value * config.hhll_tolerance_atr_multiplier
    pct_part = abs(reference_value) * config.hhll_tolerance_pct
    return max(atr_part, pct_part)


def structure_label(
    kind: str,
    candidate_value: float,
    previous_same_side_value: float | None,
    atr_value: float,
    config: BtcStructureConfig,
) -> str:
    if previous_same_side_value is None:
        return "INITIAL_HIGH" if kind == "high" else "INITIAL_LOW"
    tolerance = _label_tolerance(previous_same_side_value, atr_value, config)
    delta = candidate_value - previous_same_side_value
    if abs(delta) <= tolerance:
        return "EQH" if kind == "high" else "EQL"
    if kind == "high":
        return "HH" if delta > 0 else "LH"
    return "HL" if delta > 0 else "LL"


# ---------------------------------------------------------------------------
# Confirmation distance / excursion
# ---------------------------------------------------------------------------

def _conf_distance(candidate_value: float, atr_value: float, config: BtcStructureConfig) -> float:
    atr_part = 0.0 if math.isnan(atr_value) else atr_value * config.atr_multiplier
    pct_part = abs(candidate_value) * config.pct_threshold
    return max(atr_part, pct_part)


def _excursion_metrics(kind: str, candidate_value: float, bar: BarData) -> dict[str, float]:
    if kind == "high":
        return {"wick": candidate_value - bar.low, "close": candidate_value - bar.close}
    return {"wick": bar.high - candidate_value, "close": bar.close - candidate_value}


# ---------------------------------------------------------------------------
# Candidate / confirmed state templates
# ---------------------------------------------------------------------------

def _empty_candidate(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": None,
        "close_time": pd.NaT,
        "index": None,
        "bars_active": 0,
        "atr_at_candidate": np.nan,
        "breaks_structure": False,
        "reference_confirmed_value": None,
        "seed_reason": None,
        "confluence_windows": [],
        "confluence_count": 0,
        "confluence_short_hits": 0,
        "confluence_long_hits": 0,
    }


def _confirmed_template() -> dict[str, Any]:
    return {
        "value": None,
        "close_time": pd.NaT,
        "index": None,
        "confirmed_on": pd.NaT,
        "bars_to_confirmation": np.nan,
        "structure_label": None,
        "breaks_structure": False,
    }


# ---------------------------------------------------------------------------
# Confluence
# ---------------------------------------------------------------------------

def _candidate_confluence(
    kind: str,
    candidate_value: float,
    bar: BarData,
    windows: tuple[int, ...],
    atr_value: float,
    tolerance_atr_multiplier: float,
) -> list[int]:
    tolerance = 0.0 if math.isnan(atr_value) else atr_value * tolerance_atr_multiplier
    matches: list[int] = []
    levels = bar.rolling_highs if kind == "high" else bar.rolling_lows
    for window in windows:
        level = levels.get(window, np.nan)
        if math.isnan(level):
            continue
        if kind == "high" and candidate_value >= level - tolerance:
            matches.append(window)
        elif kind == "low" and candidate_value <= level + tolerance:
            matches.append(window)
    return matches


def _confluence_bucket_hits(
    windows: list[int],
    *,
    short_max_window: int,
    long_min_window: int,
) -> tuple[int, int]:
    short_hits = sum(1 for w in windows if w <= short_max_window)
    long_hits = sum(1 for w in windows if w >= long_min_window)
    return short_hits, long_hits


# ---------------------------------------------------------------------------
# Candidate management
# ---------------------------------------------------------------------------

def _refresh_candidate(
    kind: str,
    candidate: dict[str, Any],
    latest_confirmed: dict[str, Any],
    bar: BarData,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    refreshed = dict(candidate)
    refreshed["reference_confirmed_value"] = latest_confirmed.get("value")
    refreshed["confluence_windows"] = _candidate_confluence(
        kind,
        float(refreshed["value"]),
        bar,
        config.level_windows,
        refreshed["atr_at_candidate"],
        config.level_tolerance_atr_multiplier,
    )
    refreshed["confluence_count"] = len(refreshed["confluence_windows"])
    short_hits, long_hits = _confluence_bucket_hits(
        refreshed["confluence_windows"],
        short_max_window=config.short_confluence_max_window,
        long_min_window=config.long_confluence_min_window,
    )
    refreshed["confluence_short_hits"] = short_hits
    refreshed["confluence_long_hits"] = long_hits
    reference = refreshed["reference_confirmed_value"]
    if refreshed["index"] is None or reference is None:
        refreshed["breaks_structure"] = False
    elif kind == "high":
        refreshed["breaks_structure"] = float(refreshed["value"]) > float(reference)
    else:
        refreshed["breaks_structure"] = float(refreshed["value"]) < float(reference)
    return refreshed


def _build_candidate(
    kind: str,
    bar: BarData,
    latest_confirmed: dict[str, Any],
    seed_reason: str,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    value = bar.high if kind == "high" else bar.low
    atr = bar.atr if not math.isnan(bar.atr) else np.nan
    candidate: dict[str, Any] = {
        "kind": kind,
        "value": value,
        "close_time": pd.Timestamp(bar.close_time),
        "index": bar.index,
        "bars_active": 0,
        "atr_at_candidate": atr,
        "breaks_structure": False,
        "reference_confirmed_value": None,
        "seed_reason": seed_reason,
        "confluence_windows": [],
        "confluence_count": 0,
        "confluence_short_hits": 0,
        "confluence_long_hits": 0,
    }
    return _refresh_candidate(kind, candidate, latest_confirmed, bar, config)


def _build_candidate_from_window(
    kind: str,
    start_idx: int,
    end_idx: int,
    arrays: BarArrays,
    latest_confirmed: dict[str, Any],
    seed_reason: str,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    arr = arrays.high if kind == "high" else arrays.low
    window_slice = arr[start_idx : end_idx + 1]
    relative_idx = int(np.nanargmax(window_slice) if kind == "high" else np.nanargmin(window_slice))
    abs_idx = start_idx + relative_idx
    bar = arrays.bar_at(abs_idx)
    return _build_candidate(kind, bar, latest_confirmed, seed_reason, config)


def _candidate_is_stale(candidate: dict[str, Any], current_index: int, config: BtcStructureConfig) -> bool:
    if candidate["index"] is None:
        return False
    return (
        (current_index - int(candidate["index"])) >= config.rolling_lookback
        or int(candidate["bars_active"]) >= config.max_candidate_bars
    )


def _should_replace_candidate(
    kind: str,
    new_value: float,
    old_value: float,
    atr_value: float,
    config: BtcStructureConfig,
) -> bool:
    if kind == "high" and not (new_value > old_value):
        return False
    if kind == "low" and not (new_value < old_value):
        return False
    delta = abs(new_value - old_value)
    atr_step = 0.0 if math.isnan(atr_value) else atr_value * config.candidate_replace_min_atr_step
    pct_step = abs(old_value) * config.candidate_replace_min_pct_step
    return delta >= max(atr_step, pct_step)


# ---------------------------------------------------------------------------
# Confirmation signal
# ---------------------------------------------------------------------------

def _confirmation_signal(
    kind: str,
    bar: BarData,
    candidate: dict[str, Any],
    bars_since_candidate: int,
    config: BtcStructureConfig,
) -> tuple[bool, str, dict[str, float], float]:
    atr = bar.atr if not math.isnan(bar.atr) else candidate["atr_at_candidate"]
    threshold = _conf_distance(float(candidate["value"]), atr, config)
    metrics = _excursion_metrics(kind, float(candidate["value"]), bar)
    if metrics["close"] >= threshold:
        return True, "close", metrics, threshold
    wick_override = (
        int(candidate["confluence_count"]) >= config.level_confluence_required
        and bars_since_candidate >= (config.min_bars_confirmation + 1)
        and metrics["wick"] >= threshold * 1.15
    )
    if wick_override:
        return True, "wick_override", metrics, threshold
    return False, "none", metrics, threshold


def _peek_confirmation(
    kind: str,
    state: dict[str, Any],
    bar: BarData,
    config: BtcStructureConfig,
) -> dict[str, Any] | None:
    candidate = state[f"candidate_{kind}"]
    latest_confirmed = state[f"latest_confirmed_{kind}"]
    if candidate["index"] is None:
        return None
    bars_since_candidate = bar.index - int(candidate["index"])
    if bars_since_candidate < config.min_bars_confirmation:
        return None
    passed, mode_used, metrics, threshold = _confirmation_signal(
        kind, bar, candidate, bars_since_candidate, config,
    )
    if not passed:
        return None
    short_hits = int(candidate["confluence_short_hits"])
    long_hits = int(candidate["confluence_long_hits"])
    multi_horizon_ok = True
    if config.require_multi_horizon_confluence:
        multi_horizon_ok = (
            short_hits >= config.min_short_confluence_hits
            and long_hits >= config.min_long_confluence_hits
        )
    confluence_ok = (
        int(candidate["confluence_count"]) >= config.level_confluence_required
        and multi_horizon_ok
    ) or bool(candidate["breaks_structure"])
    if not confluence_ok and bars_since_candidate < config.force_confirmation_after_bars:
        return None
    return {
        "kind": kind,
        "available_on": pd.Timestamp(bar.close_time),
        "swing_date": pd.Timestamp(candidate["close_time"]),
        "value": float(candidate["value"]),
        "candidate_index": int(candidate["index"]),
        "bars_to_confirmation": bars_since_candidate,
        "threshold": threshold,
        "wick_excursion": metrics["wick"],
        "close_excursion": metrics["close"],
        "confirmation_price_mode_used": mode_used,
        "confirmation_path": "confluence" if confluence_ok else "forced_maturity",
        "structure_label": structure_label(
            kind,
            float(candidate["value"]),
            latest_confirmed.get("value"),
            bar.atr,
            config,
        ),
        "breaks_structure": bool(candidate["breaks_structure"]),
        "reference_confirmed_value": candidate["reference_confirmed_value"],
        "confluence_count": int(candidate["confluence_count"]),
        "confluence_short_hits": short_hits,
        "confluence_long_hits": long_hits,
        "multi_horizon_ok": multi_horizon_ok,
        "confluence_windows": ",".join(map(str, candidate["confluence_windows"])),
    }


# ---------------------------------------------------------------------------
# Event bookkeeping
# ---------------------------------------------------------------------------

def _push_candidate_event(
    stats: dict[str, Any],
    kind: str,
    candidate: dict[str, Any],
    close_time: Any,
    event_type: str,
    reason: str,
) -> None:
    stats[f"candidate_{kind}_events"].append({
        "event_type": event_type,
        "available_on": pd.Timestamp(close_time),
        "swing_date": pd.Timestamp(candidate["close_time"]),
        "value": float(candidate["value"]),
        "reason": reason,
        "breaks_structure": bool(candidate["breaks_structure"]),
        "reference_confirmed_value": candidate["reference_confirmed_value"],
        "confluence_count": int(candidate["confluence_count"]),
        "confluence_short_hits": int(candidate["confluence_short_hits"]),
        "confluence_long_hits": int(candidate["confluence_long_hits"]),
        "confluence_windows": ",".join(map(str, candidate["confluence_windows"])),
    })


def _push_break_attempt_event(
    stats: dict[str, Any],
    kind: str,
    candidate: dict[str, Any],
    close_time: Any,
) -> None:
    stats[f"break_attempt_{kind}_events"].append({
        "available_on": pd.Timestamp(close_time),
        "swing_date": pd.Timestamp(candidate["close_time"]),
        "value": float(candidate["value"]),
        "reference_confirmed_value": candidate["reference_confirmed_value"],
        "confluence_count": int(candidate["confluence_count"]),
        "confluence_short_hits": int(candidate["confluence_short_hits"]),
        "confluence_long_hits": int(candidate["confluence_long_hits"]),
        "confluence_windows": ",".join(map(str, candidate["confluence_windows"])),
        "seed_reason": candidate["seed_reason"],
    })


# ---------------------------------------------------------------------------
# Candidate update (per bar)
# ---------------------------------------------------------------------------

def _update_active_candidate(
    kind: str,
    state: dict[str, Any],
    bar: BarData,
    arrays: BarArrays,
    stats: dict[str, Any],
    config: BtcStructureConfig,
) -> dict[str, bool]:
    key = f"candidate_{kind}"
    latest_key = f"latest_confirmed_{kind}"
    replace_key = f"candidate_{kind}_replaced_before_confirmation"
    candidate = state[key]
    latest_confirmed = state[latest_key]
    previous_break = bool(candidate.get("breaks_structure", False)) if candidate["index"] is not None else False

    if candidate["index"] is None:
        state[key] = _build_candidate(kind, bar, latest_confirmed, "active_side_seed", config)
        _push_candidate_event(stats, kind, state[key], bar.close_time, f"candidate_{kind}_seeded", "active_side_seed")
        new_break = bool(state[key]["breaks_structure"])
        if new_break:
            _push_break_attempt_event(stats, kind, state[key], bar.close_time)
        return {"changed": True, "new_break": new_break}

    state[key]["bars_active"] += 1
    changed = False
    new_value = bar.high if kind == "high" else bar.low

    if _should_replace_candidate(kind, new_value, candidate["value"], bar.atr, config):
        stats[replace_key] += 1
        state[key] = _build_candidate(kind, bar, latest_confirmed, "better_extreme", config)
        _push_candidate_event(stats, kind, state[key], bar.close_time, f"candidate_{kind}_replaced", "better_extreme")
        changed = True
    elif _candidate_is_stale(candidate, bar.index, config):
        lookback_start = max(0, bar.index - config.rolling_lookback + 1)
        replacement = _build_candidate_from_window(
            kind, lookback_start, bar.index, arrays,
            latest_confirmed, "lookback_reanchor", config,
        )
        if replacement["index"] != candidate["index"] or replacement["value"] != candidate["value"]:
            stats[replace_key] += 1
            state[key] = replacement
            _push_candidate_event(stats, kind, state[key], bar.close_time, f"candidate_{kind}_reanchored", "lookback_reanchor")
            changed = True
        else:
            state[key] = _refresh_candidate(kind, state[key], latest_confirmed, bar, config)
    else:
        state[key] = _refresh_candidate(kind, state[key], latest_confirmed, bar, config)

    current_break = bool(state[key].get("breaks_structure", False))
    new_break = current_break and (changed or not previous_break)
    if new_break:
        _push_break_attempt_event(stats, kind, state[key], bar.close_time)
    return {"changed": changed, "new_break": new_break}


# ---------------------------------------------------------------------------
# Bootstrap confirmation chooser
# ---------------------------------------------------------------------------

def _choose_bootstrap_confirmation(
    high_event: dict[str, Any] | None,
    low_event: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if high_event is None and low_event is None:
        return None
    if high_event is None:
        return low_event
    if low_event is None:
        return high_event
    if high_event["candidate_index"] != low_event["candidate_index"]:
        return high_event if high_event["candidate_index"] < low_event["candidate_index"] else low_event
    high_score = (high_event["confluence_count"], -high_event["bars_to_confirmation"])
    low_score = (low_event["confluence_count"], -low_event["bars_to_confirmation"])
    return high_event if high_score >= low_score else low_event


# ---------------------------------------------------------------------------
# Apply confirmation
# ---------------------------------------------------------------------------

def _opposite_side(kind: str) -> str:
    return "low" if kind == "high" else "high"


def _apply_confirmation(
    state: dict[str, Any],
    event: dict[str, Any],
    bar: BarData,
    stats: dict[str, Any],
    config: BtcStructureConfig,
) -> None:
    kind = event["kind"]
    other = _opposite_side(kind)
    state[f"latest_confirmed_{kind}"] = {
        "value": float(event["value"]),
        "close_time": pd.Timestamp(event["swing_date"]),
        "index": int(event["candidate_index"]),
        "confirmed_on": pd.Timestamp(event["available_on"]),
        "bars_to_confirmation": int(event["bars_to_confirmation"]),
        "structure_label": event["structure_label"],
        "breaks_structure": bool(event["breaks_structure"]),
    }
    state[f"candidate_{kind}"] = _empty_candidate(kind)
    state["active_side"] = other
    state["last_confirmed_side"] = kind
    stats[f"confirmed_{kind}_events"].append(event)
    stats[f"bars_to_confirm_{kind}"].append(int(event["bars_to_confirmation"]))

    seeded = _build_candidate(other, bar, state[f"latest_confirmed_{other}"], "post_confirmation_seed", config)
    state[f"candidate_{other}"] = seeded
    _push_candidate_event(stats, other, seeded, bar.close_time, f"candidate_{other}_seeded", "post_confirmation_seed")
    if bool(seeded["breaks_structure"]):
        _push_break_attempt_event(stats, other, seeded, bar.close_time)


# ---------------------------------------------------------------------------
# Structure break detection (BOS / CHoCH) — simultaneous break bug fixed
# ---------------------------------------------------------------------------

def _break_threshold(reference_value: float, atr_value: float, config: BtcStructureConfig) -> float:
    atr_part = 0.0 if math.isnan(atr_value) else atr_value * config.bos_choch_atr_multiplier
    pct_part = abs(reference_value) * config.bos_choch_pct
    return max(atr_part, pct_part)


def _compute_structure_break_event(
    state: dict[str, Any],
    bar: BarData,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    bias_asof = state["market_bias"]
    confirmed_high = state["latest_confirmed_high"].get("value")
    confirmed_low = state["latest_confirmed_low"].get("value")
    atr_value = bar.atr if not math.isnan(bar.atr) else np.nan

    up_break = False
    down_break = False
    up_threshold = np.nan
    down_threshold = np.nan

    if confirmed_high is not None:
        up_threshold = _break_threshold(float(confirmed_high), atr_value, config)
        up_break = bar.close > float(confirmed_high) + up_threshold

    if confirmed_low is not None:
        down_threshold = _break_threshold(float(confirmed_low), atr_value, config)
        down_break = bar.close < float(confirmed_low) - down_threshold

    # When both sides break on the same bar, pick the one with larger
    # excursion relative to its threshold so bias stays consistent.
    if up_break and down_break:
        up_excursion = (bar.close - float(confirmed_high) - up_threshold) / max(up_threshold, 1e-12)
        down_excursion = (float(confirmed_low) - down_threshold - bar.close) / max(down_threshold, 1e-12)
        if up_excursion >= down_excursion:
            down_break = False
        else:
            up_break = False

    bos_up = bos_down = choch_up = choch_down = False
    broken_level_kind = None
    broken_level_value = np.nan
    threshold = np.nan
    bias_after_close = bias_asof

    if up_break:
        broken_level_kind = "high"
        broken_level_value = float(confirmed_high)
        threshold = up_threshold
        choch_up = bias_asof == "bearish"
        bos_up = not choch_up
        bias_after_close = "bullish"
    elif down_break:
        broken_level_kind = "low"
        broken_level_value = float(confirmed_low)
        threshold = down_threshold
        choch_down = bias_asof == "bullish"
        bos_down = not choch_down
        bias_after_close = "bearish"

    event_name = None
    if choch_up:
        event_name = "choch_up"
    elif choch_down:
        event_name = "choch_down"
    elif bos_up:
        event_name = "bos_up"
    elif bos_down:
        event_name = "bos_down"

    return {
        "market_bias_asof": bias_asof,
        "market_bias_after_close": bias_after_close,
        "bos_up_on_close_flag": bos_up,
        "bos_down_on_close_flag": bos_down,
        "choch_up_on_close_flag": choch_up,
        "choch_down_on_close_flag": choch_down,
        "structure_break_event": event_name,
        "structure_break_level_kind": broken_level_kind,
        "structure_break_level_value": broken_level_value,
        "structure_break_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Feature row state snapshot
# ---------------------------------------------------------------------------

def _state_feature_prefix(state: dict[str, Any], kind: str) -> dict[str, Any]:
    candidate = state[f"candidate_{kind}"]
    confirmed = state[f"latest_confirmed_{kind}"]
    return {
        f"{kind}_candidate_value_asof": candidate.get("value"),
        f"{kind}_candidate_close_time_asof": candidate.get("close_time"),
        f"{kind}_candidate_bars_active_asof": candidate.get("bars_active"),
        f"{kind}_candidate_confluence_count_asof": candidate.get("confluence_count"),
        f"{kind}_candidate_confluence_short_hits_asof": candidate.get("confluence_short_hits"),
        f"{kind}_candidate_confluence_long_hits_asof": candidate.get("confluence_long_hits"),
        f"{kind}_candidate_breaks_structure_asof": candidate.get("breaks_structure"),
        f"{kind}_confirmed_value_asof": confirmed.get("value"),
        f"{kind}_confirmed_swing_close_time_asof": confirmed.get("close_time"),
        f"{kind}_confirmed_available_on_asof": confirmed.get("confirmed_on"),
        f"{kind}_confirmed_structure_label_asof": confirmed.get("structure_label"),
        f"{kind}_confirmed_breaks_structure_asof": confirmed.get("breaks_structure"),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_structure(
    config: BtcStructureConfig,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    confirmed_highs: pd.DataFrame,
    confirmed_lows: pd.DataFrame,
    structure_breaks: pd.DataFrame,
    stats: dict[str, Any],
) -> dict[str, Any]:
    high_counts = (
        confirmed_highs["structure_label"].value_counts(dropna=False).sort_index().to_dict()
        if not confirmed_highs.empty else {}
    )
    low_counts = (
        confirmed_lows["structure_label"].value_counts(dropna=False).sort_index().to_dict()
        if not confirmed_lows.empty else {}
    )
    break_counts = (
        structure_breaks["event"].value_counts(dropna=False).sort_index().to_dict()
        if not structure_breaks.empty else {}
    )
    latest_feature = features.iloc[-1]
    return {
        "interval": config.interval,
        "market_type": config.market_type,
        "bars": len(ohlcv),
        "start": pd.Timestamp(ohlcv["close_time"].iloc[0]).isoformat(),
        "end": pd.Timestamp(ohlcv["close_time"].iloc[-1]).isoformat(),
        "latest_close": float(ohlcv["close"].iloc[-1]),
        "confirmed_highs": len(confirmed_highs),
        "confirmed_lows": len(confirmed_lows),
        "avg_bars_to_confirm_high": float(np.mean(stats["bars_to_confirm_high"])) if stats["bars_to_confirm_high"] else None,
        "avg_bars_to_confirm_low": float(np.mean(stats["bars_to_confirm_low"])) if stats["bars_to_confirm_low"] else None,
        "candidate_high_replaced_before_confirmation": stats["candidate_high_replaced_before_confirmation"],
        "candidate_low_replaced_before_confirmation": stats["candidate_low_replaced_before_confirmation"],
        "high_label_counts": {str(k): int(v) for k, v in high_counts.items()},
        "low_label_counts": {str(k): int(v) for k, v in low_counts.items()},
        "structure_break_counts": {str(k): int(v) for k, v in break_counts.items()},
        "last_market_bias_after_close": latest_feature["market_bias_after_close"],
        "last_confirmed_high_value_asof": latest_feature.get("high_confirmed_value_asof"),
        "last_confirmed_low_value_asof": latest_feature.get("low_confirmed_value_asof"),
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_btc_structure(
    ohlcv: pd.DataFrame,
    config: BtcStructureConfig,
    *,
    checkpoint: StructureCheckpoint | None = None,
) -> tuple[StructureArtifacts, StructureCheckpoint]:
    work = ohlcv.copy()
    work["atr"] = causal_atr(work, config.atr_window)
    work = _build_rolling_levels(work, config.level_windows)
    arrays = _extract_arrays(work, config)

    # -- restore from checkpoint or initialise fresh -----------------------
    fingerprint = repr(config)
    can_resume = (
        checkpoint is not None
        and checkpoint.resume_from <= arrays.length
        and checkpoint.config_fingerprint == fingerprint
        and checkpoint.resume_from > 0
        and _ohlcv_prefix_hash(ohlcv, checkpoint.resume_from) == checkpoint.prefix_hash
    )

    if can_resume:
        assert checkpoint is not None  # for type narrowing
        state = copy.deepcopy(checkpoint.state)
        stats = copy.deepcopy(checkpoint.stats)
        feature_rows: list[dict[str, Any]] = list(checkpoint.feature_rows)
        start_idx = checkpoint.resume_from
    else:
        state: dict[str, Any] = {
            "candidate_high": _empty_candidate("high"),
            "candidate_low": _empty_candidate("low"),
            "latest_confirmed_high": _confirmed_template(),
            "latest_confirmed_low": _confirmed_template(),
            "active_side": None,
            "last_confirmed_side": None,
            "market_bias": "neutral",
        }
        stats: dict[str, Any] = {
            "candidate_high_events": [],
            "candidate_low_events": [],
            "break_attempt_high_events": [],
            "break_attempt_low_events": [],
            "confirmed_high_events": [],
            "confirmed_low_events": [],
            "bars_to_confirm_high": [],
            "bars_to_confirm_low": [],
            "candidate_high_replaced_before_confirmation": 0,
            "candidate_low_replaced_before_confirmation": 0,
            "structure_break_events": [],
        }
        feature_rows: list[dict[str, Any]] = []
        start_idx = 0

    # Reuse a single mutable BarData across the hot loop to avoid
    # allocating a new object + two dicts per bar.
    bar = BarData(rolling_highs={w: 0.0 for w in config.level_windows},
                  rolling_lows={w: 0.0 for w in config.level_windows})

    for i in range(start_idx, arrays.length):
        arrays.fill_bar(i, bar)
        break_event = _compute_structure_break_event(state, bar, config)

        if break_event["structure_break_event"] is not None:
            stats["structure_break_events"].append({
                "available_on": pd.Timestamp(bar.close_time),
                "event": break_event["structure_break_event"],
                "level_kind": break_event["structure_break_level_kind"],
                "level_value": break_event["structure_break_level_value"],
                "threshold": break_event["structure_break_threshold"],
                "market_bias_asof": break_event["market_bias_asof"],
                "market_bias_after_close": break_event["market_bias_after_close"],
                "close": bar.close,
            })

        feature_row: dict[str, Any] = {
            "close_time": pd.Timestamp(bar.close_time),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "atr": bar.atr if not math.isnan(bar.atr) else np.nan,
            "active_side_asof": state["active_side"],
            "last_confirmed_side_asof": state["last_confirmed_side"],
            "market_bias_asof": break_event["market_bias_asof"],
            "market_bias_after_close": break_event["market_bias_after_close"],
            "bos_up_on_close_flag": break_event["bos_up_on_close_flag"],
            "bos_down_on_close_flag": break_event["bos_down_on_close_flag"],
            "choch_up_on_close_flag": break_event["choch_up_on_close_flag"],
            "choch_down_on_close_flag": break_event["choch_down_on_close_flag"],
            "choch_any_on_close_flag": break_event["choch_up_on_close_flag"] or break_event["choch_down_on_close_flag"],
            "bos_any_on_close_flag": break_event["bos_up_on_close_flag"] or break_event["bos_down_on_close_flag"],
            "structure_break_event_on_close": break_event["structure_break_event"],
            "structure_break_level_kind_on_close": break_event["structure_break_level_kind"],
            "structure_break_level_value_on_close": break_event["structure_break_level_value"],
            "structure_break_threshold_on_close": break_event["structure_break_threshold"],
            "high_break_attempt_on_close_flag": False,
            "low_break_attempt_on_close_flag": False,
            "confirmed_high_on_close_flag": False,
            "confirmed_low_on_close_flag": False,
            "confirmed_high_label_on_close": None,
            "confirmed_low_label_on_close": None,
        }
        feature_row.update(_state_feature_prefix(state, "high"))
        feature_row.update(_state_feature_prefix(state, "low"))

        if state["active_side"] is None:
            high_update = _update_active_candidate("high", state, bar, arrays, stats, config)
            low_update = _update_active_candidate("low", state, bar, arrays, stats, config)
            feature_row["high_break_attempt_on_close_flag"] = bool(high_update["new_break"])
            feature_row["low_break_attempt_on_close_flag"] = bool(low_update["new_break"])
            high_event = _peek_confirmation("high", state, bar, config)
            low_event = _peek_confirmation("low", state, bar, config)
            chosen_event = _choose_bootstrap_confirmation(high_event, low_event)
            if chosen_event is not None:
                _apply_confirmation(state, chosen_event, bar, stats, config)
                if chosen_event["kind"] == "high":
                    feature_row["confirmed_high_on_close_flag"] = True
                    feature_row["confirmed_high_label_on_close"] = chosen_event["structure_label"]
                else:
                    feature_row["confirmed_low_on_close_flag"] = True
                    feature_row["confirmed_low_label_on_close"] = chosen_event["structure_label"]
        else:
            active_kind = str(state["active_side"])
            active_update = _update_active_candidate(active_kind, state, bar, arrays, stats, config)
            feature_row[f"{active_kind}_break_attempt_on_close_flag"] = bool(active_update["new_break"])
            chosen_event = _peek_confirmation(active_kind, state, bar, config)
            if chosen_event is not None:
                _apply_confirmation(state, chosen_event, bar, stats, config)
                if active_kind == "high":
                    feature_row["confirmed_high_on_close_flag"] = True
                    feature_row["confirmed_high_label_on_close"] = chosen_event["structure_label"]
                else:
                    feature_row["confirmed_low_on_close_flag"] = True
                    feature_row["confirmed_low_label_on_close"] = chosen_event["structure_label"]

        state["market_bias"] = break_event["market_bias_after_close"]
        feature_rows.append(feature_row)

    # -- build checkpoint ----------------------------------------------------
    new_checkpoint = StructureCheckpoint(
        resume_from=arrays.length,
        state=copy.deepcopy(state),
        stats=copy.deepcopy(stats),
        feature_rows=list(feature_rows),
        prefix_hash=_ohlcv_prefix_hash(ohlcv, arrays.length),
        config_fingerprint=fingerprint,
    )

    # -- build artifacts ----------------------------------------------------
    features = pd.DataFrame(feature_rows)
    candidate_highs = pd.DataFrame(stats["candidate_high_events"])
    candidate_lows = pd.DataFrame(stats["candidate_low_events"])
    confirmed_highs = pd.DataFrame(stats["confirmed_high_events"])
    confirmed_lows = pd.DataFrame(stats["confirmed_low_events"])
    break_attempt_highs = pd.DataFrame(stats["break_attempt_high_events"])
    break_attempt_lows = pd.DataFrame(stats["break_attempt_low_events"])
    structure_breaks = pd.DataFrame(stats["structure_break_events"])
    summary = summarize_structure(config, ohlcv, features, confirmed_highs, confirmed_lows, structure_breaks, stats)

    artifacts = StructureArtifacts(
        ohlcv=ohlcv,
        features=features,
        candidate_highs=candidate_highs,
        candidate_lows=candidate_lows,
        confirmed_highs=confirmed_highs,
        confirmed_lows=confirmed_lows,
        break_attempt_highs=break_attempt_highs,
        break_attempt_lows=break_attempt_lows,
        structure_breaks=structure_breaks,
        summary=summary,
    )
    return artifacts, new_checkpoint
