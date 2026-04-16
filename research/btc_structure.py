from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtester.data import BinanceClient
from backtester.models import MarketType
from backtester.preview import interval_to_seconds

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional visualization dependency
    go = None


_BASE_LEVEL_DAYS = (1, 3, 7, 10, 30, 90, 180, 300, 365)


@dataclass(slots=True, frozen=True)
class BtcStructureConfig:
    ticker: str = "BTC/USDT"
    market_type: MarketType = MarketType.FUTURES
    interval: str = "1d"
    years: int = 5

    rolling_lookback: int = 400
    atr_window: int = 14
    atr_multiplier: float = 1.25
    pct_threshold: float = 0.015
    min_bars_confirmation: int = 3
    force_confirmation_after_bars: int = 7
    max_candidate_bars: int = 18

    level_windows: tuple[int, ...] = _BASE_LEVEL_DAYS
    level_confluence_required: int = 2
    level_tolerance_atr_multiplier: float = 0.50
    require_multi_horizon_confluence: bool = True
    short_confluence_max_window: int = 30
    long_confluence_min_window: int = 90
    min_short_confluence_hits: int = 1
    min_long_confluence_hits: int = 1

    candidate_replace_min_atr_step: float = 0.10
    candidate_replace_min_pct_step: float = 0.001

    hhll_tolerance_atr_multiplier: float = 0.15
    hhll_tolerance_pct: float = 0.001
    bos_choch_atr_multiplier: float = 0.35
    bos_choch_pct: float = 0.003

    @staticmethod
    def for_interval(
        interval: str,
        *,
        ticker: str = "BTC/USDT",
        market_type: MarketType = MarketType.FUTURES,
        years: int = 5,
    ) -> "BtcStructureConfig":
        seconds = interval_to_seconds(interval)
        bars_per_day = max(1, int(round(86400 / seconds)))

        def bars(days: int) -> int:
            return max(1, int(round(days * bars_per_day)))

        return BtcStructureConfig(
            ticker=ticker,
            market_type=market_type,
            interval=interval,
            years=years,
            rolling_lookback=max(bars(400), bars(365) + bars(30)),
            atr_window=bars(14),
            min_bars_confirmation=max(2, bars(3)),
            force_confirmation_after_bars=max(bars(7), bars(3) + 2),
            max_candidate_bars=max(bars(18), bars(7) + bars(3)),
            level_windows=tuple(sorted({bars(days) for days in _BASE_LEVEL_DAYS})),
            short_confluence_max_window=bars(30),
            long_confluence_min_window=bars(90),
        )


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


def fetch_btc_ohlcv(
    config: BtcStructureConfig,
    *,
    end: datetime | None = None,
    client: BinanceClient | None = None,
) -> pd.DataFrame:
    end_dt = end or datetime.now(UTC)
    start_dt = end_dt - timedelta(days=365 * config.years)
    active_client = client or BinanceClient(market_type=config.market_type)
    candles = active_client.fetch_klines(
        config.ticker,
        config.interval,
        start_dt,
        end_dt,
    )
    rows = [
        {
            "date": candle.close_time,
            "ticker": config.ticker,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no OHLCV data returned for {config.ticker} {config.interval}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame


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


def _label_tolerance(reference_value: float, atr_value: float, config: BtcStructureConfig) -> float:
    atr_part = 0.0 if pd.isna(atr_value) else float(atr_value) * config.hhll_tolerance_atr_multiplier
    pct_part = abs(float(reference_value)) * config.hhll_tolerance_pct
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
    delta = float(candidate_value) - float(previous_same_side_value)
    if abs(delta) <= tolerance:
        return "EQH" if kind == "high" else "EQL"
    if kind == "high":
        return "HH" if delta > 0 else "LH"
    return "HL" if delta > 0 else "LL"


def _conf_distance(candidate_value: float, atr_value: float, config: BtcStructureConfig) -> float:
    atr_part = 0.0 if pd.isna(atr_value) else float(atr_value) * config.atr_multiplier
    pct_part = abs(float(candidate_value)) * config.pct_threshold
    return max(atr_part, pct_part)


def _empty_candidate(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": None,
        "date": pd.NaT,
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
        "date": pd.NaT,
        "index": None,
        "confirmed_on": pd.NaT,
        "bars_to_confirmation": np.nan,
        "structure_label": None,
        "breaks_structure": False,
    }


def _rolling_slice(frame: pd.DataFrame, end_index: int, lookback: int) -> pd.DataFrame:
    start = max(0, end_index - lookback + 1)
    return frame.iloc[start : end_index + 1]


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


def _candidate_confluence(
    kind: str,
    candidate_value: float,
    row: pd.Series,
    windows: tuple[int, ...],
    atr_value: float,
    tolerance_atr_multiplier: float,
) -> list[int]:
    tolerance = 0.0 if pd.isna(atr_value) else float(atr_value) * tolerance_atr_multiplier
    matches: list[int] = []
    for window in windows:
        column = f"rolling_high_{window}" if kind == "high" else f"rolling_low_{window}"
        level = row.get(column, np.nan)
        if pd.isna(level):
            continue
        if kind == "high" and float(candidate_value) >= float(level) - tolerance:
            matches.append(int(window))
        if kind == "low" and float(candidate_value) <= float(level) + tolerance:
            matches.append(int(window))
    return matches


def _confluence_bucket_hits(
    windows: list[int],
    *,
    short_max_window: int,
    long_min_window: int,
) -> tuple[int, int]:
    short_hits = sum(1 for window in windows if int(window) <= int(short_max_window))
    long_hits = sum(1 for window in windows if int(window) >= int(long_min_window))
    return short_hits, long_hits


def _refresh_candidate(
    kind: str,
    candidate: dict[str, Any],
    latest_confirmed: dict[str, Any],
    row: pd.Series,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    refreshed = dict(candidate)
    refreshed["reference_confirmed_value"] = latest_confirmed.get("value")
    refreshed["confluence_windows"] = _candidate_confluence(
        kind,
        float(refreshed["value"]),
        row,
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
    row: pd.Series,
    index: int,
    latest_confirmed: dict[str, Any],
    seed_reason: str,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    field = "high" if kind == "high" else "low"
    candidate = {
        "kind": kind,
        "value": float(row[field]),
        "date": pd.Timestamp(row["date"]),
        "index": int(index),
        "bars_active": 0,
        "atr_at_candidate": float(row["atr"]) if not pd.isna(row["atr"]) else np.nan,
        "breaks_structure": False,
        "reference_confirmed_value": None,
        "seed_reason": seed_reason,
        "confluence_windows": [],
        "confluence_count": 0,
        "confluence_short_hits": 0,
        "confluence_long_hits": 0,
    }
    return _refresh_candidate(kind, candidate, latest_confirmed, row, config)


def _build_candidate_from_window(
    kind: str,
    window: pd.DataFrame,
    latest_confirmed: dict[str, Any],
    seed_reason: str,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    field = "high" if kind == "high" else "low"
    index = window[field].idxmax() if kind == "high" else window[field].idxmin()
    row = window.loc[index]
    return _build_candidate(kind, row, int(index), latest_confirmed, seed_reason, config)


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
    if kind == "high" and not (float(new_value) > float(old_value)):
        return False
    if kind == "low" and not (float(new_value) < float(old_value)):
        return False
    delta = abs(float(new_value) - float(old_value))
    atr_step = 0.0 if pd.isna(atr_value) else float(atr_value) * config.candidate_replace_min_atr_step
    pct_step = abs(float(old_value)) * config.candidate_replace_min_pct_step
    return delta >= max(atr_step, pct_step)


def _excursion_metrics(kind: str, candidate_value: float, row: pd.Series) -> dict[str, float]:
    if kind == "high":
        wick = float(candidate_value - row["low"])
        close = float(candidate_value - row["close"])
    else:
        wick = float(row["high"] - candidate_value)
        close = float(row["close"] - candidate_value)
    return {"wick": wick, "close": close}


def _confirmation_signal(
    kind: str,
    row: pd.Series,
    candidate: dict[str, Any],
    bars_since_candidate: int,
    config: BtcStructureConfig,
) -> tuple[bool, str, dict[str, float], float]:
    threshold = _conf_distance(
        float(candidate["value"]),
        row["atr"] if not pd.isna(row["atr"]) else candidate["atr_at_candidate"],
        config,
    )
    metrics = _excursion_metrics(kind, float(candidate["value"]), row)
    close_passed = metrics["close"] >= threshold
    if close_passed:
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
    row: pd.Series,
    index: int,
    config: BtcStructureConfig,
) -> dict[str, Any] | None:
    candidate = state[f"candidate_{kind}"]
    latest_confirmed = state[f"latest_confirmed_{kind}"]
    if candidate["index"] is None:
        return None

    bars_since_candidate = index - int(candidate["index"])
    if bars_since_candidate < config.min_bars_confirmation:
        return None

    passed, mode_used, metrics, threshold = _confirmation_signal(
        kind,
        row,
        candidate,
        bars_since_candidate,
        config,
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
    matured_enough = bars_since_candidate >= config.force_confirmation_after_bars
    if not confluence_ok and not matured_enough:
        return None

    path = "confluence" if confluence_ok else "forced_maturity"
    return {
        "kind": kind,
        "available_on": pd.Timestamp(row["date"]),
        "swing_date": pd.Timestamp(candidate["date"]),
        "value": float(candidate["value"]),
        "candidate_index": int(candidate["index"]),
        "bars_to_confirmation": int(bars_since_candidate),
        "threshold": float(threshold),
        "wick_excursion": float(metrics["wick"]),
        "close_excursion": float(metrics["close"]),
        "confirmation_price_mode_used": mode_used,
        "confirmation_path": path,
        "structure_label": structure_label(
            kind,
            float(candidate["value"]),
            latest_confirmed.get("value"),
            row["atr"],
            config,
        ),
        "breaks_structure": bool(candidate["breaks_structure"]),
        "reference_confirmed_value": candidate["reference_confirmed_value"],
        "confluence_count": int(candidate["confluence_count"]),
        "confluence_short_hits": short_hits,
        "confluence_long_hits": long_hits,
        "multi_horizon_ok": bool(multi_horizon_ok),
        "confluence_windows": ",".join(map(str, candidate["confluence_windows"])),
    }


def _opposite_side(kind: str) -> str:
    return "low" if kind == "high" else "high"


def _push_candidate_event(
    stats: dict[str, Any],
    kind: str,
    candidate: dict[str, Any],
    available_on: pd.Timestamp,
    event_type: str,
    reason: str,
) -> None:
    stats[f"candidate_{kind}_events"].append(
        {
            "event_type": event_type,
            "available_on": pd.Timestamp(available_on),
            "swing_date": pd.Timestamp(candidate["date"]),
            "value": float(candidate["value"]),
            "reason": reason,
            "breaks_structure": bool(candidate["breaks_structure"]),
            "reference_confirmed_value": candidate["reference_confirmed_value"],
            "confluence_count": int(candidate["confluence_count"]),
            "confluence_short_hits": int(candidate["confluence_short_hits"]),
            "confluence_long_hits": int(candidate["confluence_long_hits"]),
            "confluence_windows": ",".join(map(str, candidate["confluence_windows"])),
        }
    )


def _push_break_attempt_event(
    stats: dict[str, Any],
    kind: str,
    candidate: dict[str, Any],
    available_on: pd.Timestamp,
) -> None:
    stats[f"break_attempt_{kind}_events"].append(
        {
            "available_on": pd.Timestamp(available_on),
            "swing_date": pd.Timestamp(candidate["date"]),
            "value": float(candidate["value"]),
            "reference_confirmed_value": candidate["reference_confirmed_value"],
            "confluence_count": int(candidate["confluence_count"]),
            "confluence_short_hits": int(candidate["confluence_short_hits"]),
            "confluence_long_hits": int(candidate["confluence_long_hits"]),
            "confluence_windows": ",".join(map(str, candidate["confluence_windows"])),
            "seed_reason": candidate["seed_reason"],
        }
    )


def _update_active_candidate(
    kind: str,
    state: dict[str, Any],
    row: pd.Series,
    index: int,
    work: pd.DataFrame,
    stats: dict[str, Any],
    config: BtcStructureConfig,
) -> dict[str, bool]:
    key = f"candidate_{kind}"
    latest_key = f"latest_confirmed_{kind}"
    replace_key = f"candidate_{kind}_replaced_before_confirmation"
    field = "high" if kind == "high" else "low"

    candidate = state[key]
    latest_confirmed = state[latest_key]
    previous_break = bool(candidate.get("breaks_structure", False)) if candidate["index"] is not None else False

    if candidate["index"] is None:
        state[key] = _build_candidate(kind, row, index, latest_confirmed, "active_side_seed", config)
        _push_candidate_event(stats, kind, state[key], row["date"], f"candidate_{kind}_seeded", "active_side_seed")
        new_break = bool(state[key]["breaks_structure"])
        if new_break:
            _push_break_attempt_event(stats, kind, state[key], row["date"])
        return {"changed": True, "new_break": new_break}

    state[key]["bars_active"] += 1
    changed = False
    current_atr = float(row["atr"]) if not pd.isna(row["atr"]) else np.nan

    if _should_replace_candidate(kind, row[field], candidate["value"], current_atr, config):
        stats[replace_key] += 1
        state[key] = _build_candidate(kind, row, index, latest_confirmed, "better_extreme", config)
        _push_candidate_event(stats, kind, state[key], row["date"], f"candidate_{kind}_replaced", "better_extreme")
        changed = True
    elif _candidate_is_stale(candidate, index, config):
        replacement = _build_candidate_from_window(
            kind,
            _rolling_slice(work, index, config.rolling_lookback),
            latest_confirmed,
            "lookback_reanchor",
            config,
        )
        if replacement["index"] != candidate["index"] or replacement["value"] != candidate["value"]:
            stats[replace_key] += 1
            state[key] = replacement
            _push_candidate_event(stats, kind, state[key], row["date"], f"candidate_{kind}_reanchored", "lookback_reanchor")
            changed = True
        else:
            state[key] = _refresh_candidate(kind, state[key], latest_confirmed, row, config)
    else:
        state[key] = _refresh_candidate(kind, state[key], latest_confirmed, row, config)

    current_break = bool(state[key].get("breaks_structure", False))
    new_break = current_break and (changed or not previous_break)
    if new_break:
        _push_break_attempt_event(stats, kind, state[key], row["date"])
    return {"changed": changed, "new_break": new_break}


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


def _apply_confirmation(
    state: dict[str, Any],
    event: dict[str, Any],
    row: pd.Series,
    index: int,
    stats: dict[str, Any],
    config: BtcStructureConfig,
) -> tuple[str, bool]:
    kind = event["kind"]
    other = _opposite_side(kind)
    state[f"latest_confirmed_{kind}"] = {
        "value": float(event["value"]),
        "date": pd.Timestamp(event["swing_date"]),
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

    seeded = _build_candidate(other, row, index, state[f"latest_confirmed_{other}"], "post_confirmation_seed", config)
    state[f"candidate_{other}"] = seeded
    _push_candidate_event(stats, other, seeded, row["date"], f"candidate_{other}_seeded", "post_confirmation_seed")
    seeded_break = bool(seeded["breaks_structure"])
    if seeded_break:
        _push_break_attempt_event(stats, other, seeded, row["date"])
    return other, seeded_break


def _break_threshold(reference_value: float, atr_value: float, config: BtcStructureConfig) -> float:
    atr_part = 0.0 if pd.isna(atr_value) else float(atr_value) * config.bos_choch_atr_multiplier
    pct_part = abs(float(reference_value)) * config.bos_choch_pct
    return max(atr_part, pct_part)


def _compute_structure_break_event(
    state: dict[str, Any],
    row: pd.Series,
    config: BtcStructureConfig,
) -> dict[str, Any]:
    bias_asof = state["market_bias"]
    confirmed_high = state["latest_confirmed_high"].get("value")
    confirmed_low = state["latest_confirmed_low"].get("value")
    atr_value = float(row["atr"]) if not pd.isna(row["atr"]) else np.nan

    bos_up = False
    bos_down = False
    choch_up = False
    choch_down = False
    broken_level_kind = None
    broken_level_value = np.nan
    threshold = np.nan
    bias_after_close = bias_asof

    if confirmed_high is not None:
        up_threshold = _break_threshold(float(confirmed_high), atr_value, config)
        if float(row["close"]) > float(confirmed_high) + up_threshold:
            broken_level_kind = "high"
            broken_level_value = float(confirmed_high)
            threshold = up_threshold
            if bias_asof == "bearish":
                choch_up = True
            else:
                bos_up = True
            bias_after_close = "bullish"

    if confirmed_low is not None:
        down_threshold = _break_threshold(float(confirmed_low), atr_value, config)
        if float(row["close"]) < float(confirmed_low) - down_threshold:
            if broken_level_kind is None:
                broken_level_kind = "low"
                broken_level_value = float(confirmed_low)
                threshold = down_threshold
            if bias_asof == "bullish":
                choch_down = True
            else:
                bos_down = True
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


def _serialize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return None if pd.isna(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _parse_confluence_windows(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    windows: list[int] = []
    for part in parts:
        try:
            windows.append(int(part))
        except ValueError:
            continue
    return windows


def _classify_level_scope(max_window: int) -> str:
    if int(max_window) >= 180:
        return "global"
    if int(max_window) >= 90:
        return "major"
    if int(max_window) >= 30:
        return "structural"
    return "local"


def _window_score(max_window: int) -> int:
    if int(max_window) >= 365:
        return 5
    if int(max_window) >= 180:
        return 4
    if int(max_window) >= 90:
        return 3
    if int(max_window) >= 30:
        return 2
    if int(max_window) >= 7:
        return 1
    return 0


def rank_confirmed_levels(confirmed_df: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    if confirmed_df.empty:
        return confirmed_df.copy()

    ranked = confirmed_df.copy()
    ranked["available_on"] = pd.to_datetime(ranked["available_on"], utc=True)
    ranked["swing_date"] = pd.to_datetime(ranked["swing_date"], utc=True)
    ranked["confluence_window_list"] = ranked["confluence_windows"].apply(_parse_confluence_windows)
    ranked["max_window"] = ranked["confluence_window_list"].apply(lambda items: max(items) if items else 0)
    ranked["mean_window"] = ranked["confluence_window_list"].apply(
        lambda items: float(np.mean(items)) if items else 0.0
    )
    ranked["level_scope"] = ranked["max_window"].apply(_classify_level_scope)
    ranked["window_score"] = ranked["max_window"].apply(_window_score)
    ranked["structure_bonus"] = ranked["breaks_structure"].fillna(False).astype(int) * 2
    ranked["label_bonus"] = ranked["structure_label"].map(
        {
            "HH": 2 if kind == "high" else 0,
            "LL": 2 if kind == "low" else 0,
            "HL": 1 if kind == "low" else 0,
            "LH": 1 if kind == "high" else 0,
            "EQH": 1 if kind == "high" else 0,
            "EQL": 1 if kind == "low" else 0,
        }
    ).fillna(0)
    ranked["level_score"] = (
        ranked["confluence_count"].fillna(0).astype(int)
        + ranked["window_score"].astype(int)
        + ranked["structure_bonus"].astype(int)
        + ranked["label_bonus"].astype(int)
    )
    ranked["level_priority"] = pd.cut(
        ranked["level_score"],
        bins=[-1, 5, 8, 11, 100],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    ranked["is_major_level"] = ranked["level_scope"].isin(["major", "global"])
    ranked["is_strategy_level"] = ranked["level_scope"].isin(["structural", "major", "global"]) & (
        ranked["level_score"] >= 7
    )
    return ranked


def rank_structure_breaks(
    structure_breaks: pd.DataFrame,
    confirmed_highs: pd.DataFrame,
    confirmed_lows: pd.DataFrame,
) -> pd.DataFrame:
    if structure_breaks.empty:
        return structure_breaks.copy()

    ranked = structure_breaks.copy()
    ranked["available_on"] = pd.to_datetime(ranked["available_on"], utc=True)

    ranked_highs = rank_confirmed_levels(confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(confirmed_lows, kind="low")

    level_scope: list[str] = []
    level_score: list[float] = []
    level_priority: list[str] = []
    level_max_window: list[int] = []

    for _, row in ranked.iterrows():
        kind = row.get("level_kind")
        level_value = row.get("level_value")
        when = pd.Timestamp(row["available_on"])
        source = ranked_highs if kind == "high" else ranked_lows
        if source.empty or pd.isna(level_value):
            level_scope.append("unknown")
            level_score.append(np.nan)
            level_priority.append("unranked")
            level_max_window.append(0)
            continue
        matches = source[
            np.isclose(source["value"].astype(float), float(level_value), rtol=0.0, atol=1e-9)
            & (source["available_on"] <= when)
        ].sort_values("available_on")
        if matches.empty:
            level_scope.append("unknown")
            level_score.append(np.nan)
            level_priority.append("unranked")
            level_max_window.append(0)
            continue
        match = matches.iloc[-1]
        level_scope.append(str(match["level_scope"]))
        level_score.append(float(match["level_score"]))
        level_priority.append(str(match["level_priority"]))
        level_max_window.append(int(match["max_window"]))

    ranked["broken_level_scope"] = level_scope
    ranked["broken_level_score"] = level_score
    ranked["broken_level_priority"] = level_priority
    ranked["broken_level_max_window"] = level_max_window
    ranked["is_major_break"] = ranked["broken_level_scope"].isin(["major", "global"])
    ranked["is_strategy_break"] = ranked["event"].astype(str).str.startswith("choch") | (
        ranked["event"].astype(str).str.startswith("bos") & ranked["is_major_break"]
    )
    return ranked


def _state_feature_prefix(state: dict[str, Any], kind: str) -> dict[str, Any]:
    candidate = state[f"candidate_{kind}"]
    confirmed = state[f"latest_confirmed_{kind}"]
    prefix = f"{kind}"
    return {
        f"{prefix}_candidate_value_asof": candidate.get("value"),
        f"{prefix}_candidate_date_asof": candidate.get("date"),
        f"{prefix}_candidate_bars_active_asof": candidate.get("bars_active"),
        f"{prefix}_candidate_confluence_count_asof": candidate.get("confluence_count"),
        f"{prefix}_candidate_confluence_short_hits_asof": candidate.get("confluence_short_hits"),
        f"{prefix}_candidate_confluence_long_hits_asof": candidate.get("confluence_long_hits"),
        f"{prefix}_candidate_breaks_structure_asof": candidate.get("breaks_structure"),
        f"{prefix}_confirmed_value_asof": confirmed.get("value"),
        f"{prefix}_confirmed_swing_date_asof": confirmed.get("date"),
        f"{prefix}_confirmed_available_on_asof": confirmed.get("confirmed_on"),
        f"{prefix}_confirmed_structure_label_asof": confirmed.get("structure_label"),
        f"{prefix}_confirmed_breaks_structure_asof": confirmed.get("breaks_structure"),
    }


def simulate_btc_structure(
    ohlcv: pd.DataFrame,
    config: BtcStructureConfig,
) -> StructureArtifacts:
    work = ohlcv.copy()
    work["atr"] = causal_atr(work, config.atr_window)
    work = _build_rolling_levels(work, config.level_windows)

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
    for index, row in work.iterrows():
        row = row.copy()
        break_event = _compute_structure_break_event(state, row, config)
        if break_event["structure_break_event"] is not None:
            stats["structure_break_events"].append(
                {
                    "available_on": pd.Timestamp(row["date"]),
                    "event": break_event["structure_break_event"],
                    "level_kind": break_event["structure_break_level_kind"],
                    "level_value": break_event["structure_break_level_value"],
                    "threshold": break_event["structure_break_threshold"],
                    "market_bias_asof": break_event["market_bias_asof"],
                    "market_bias_after_close": break_event["market_bias_after_close"],
                    "close": float(row["close"]),
                }
            )

        feature_row: dict[str, Any] = {
            "date": pd.Timestamp(row["date"]),
            "ticker": row["ticker"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "atr": float(row["atr"]) if not pd.isna(row["atr"]) else np.nan,
            "active_side_asof": state["active_side"],
            "last_confirmed_side_asof": state["last_confirmed_side"],
            "market_bias_asof": break_event["market_bias_asof"],
            "market_bias_after_close": break_event["market_bias_after_close"],
            "bos_up_on_close_flag": break_event["bos_up_on_close_flag"],
            "bos_down_on_close_flag": break_event["bos_down_on_close_flag"],
            "choch_up_on_close_flag": break_event["choch_up_on_close_flag"],
            "choch_down_on_close_flag": break_event["choch_down_on_close_flag"],
            "choch_any_on_close_flag": (
                break_event["choch_up_on_close_flag"] or break_event["choch_down_on_close_flag"]
            ),
            "bos_any_on_close_flag": (
                break_event["bos_up_on_close_flag"] or break_event["bos_down_on_close_flag"]
            ),
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
            "confirmed_high_value_on_close": np.nan,
            "confirmed_low_value_on_close": np.nan,
            "confirmed_high_swing_date_on_close": pd.NaT,
            "confirmed_low_swing_date_on_close": pd.NaT,
            "confirmed_high_bars_to_confirmation_on_close": np.nan,
            "confirmed_low_bars_to_confirmation_on_close": np.nan,
            "confirmation_event_on_close": None,
        }
        feature_row.update(_state_feature_prefix(state, "high"))
        feature_row.update(_state_feature_prefix(state, "low"))

        if state["active_side"] is None:
            high_update = _update_active_candidate("high", state, row, int(index), work, stats, config)
            low_update = _update_active_candidate("low", state, row, int(index), work, stats, config)
            feature_row["high_break_attempt_on_close_flag"] = bool(high_update["new_break"])
            feature_row["low_break_attempt_on_close_flag"] = bool(low_update["new_break"])

            high_event = _peek_confirmation("high", state, row, int(index), config)
            low_event = _peek_confirmation("low", state, row, int(index), config)
            chosen_event = _choose_bootstrap_confirmation(high_event, low_event)
            if chosen_event is not None:
                _apply_confirmation(state, chosen_event, row, int(index), stats, config)
                if chosen_event["kind"] == "high":
                    feature_row["confirmed_high_on_close_flag"] = True
                    feature_row["confirmed_high_label_on_close"] = chosen_event["structure_label"]
                    feature_row["confirmed_high_value_on_close"] = chosen_event["value"]
                    feature_row["confirmed_high_swing_date_on_close"] = chosen_event["swing_date"]
                    feature_row["confirmed_high_bars_to_confirmation_on_close"] = chosen_event[
                        "bars_to_confirmation"
                    ]
                else:
                    feature_row["confirmed_low_on_close_flag"] = True
                    feature_row["confirmed_low_label_on_close"] = chosen_event["structure_label"]
                    feature_row["confirmed_low_value_on_close"] = chosen_event["value"]
                    feature_row["confirmed_low_swing_date_on_close"] = chosen_event["swing_date"]
                    feature_row["confirmed_low_bars_to_confirmation_on_close"] = chosen_event[
                        "bars_to_confirmation"
                    ]
                feature_row["confirmation_event_on_close"] = f"confirmed_{chosen_event['kind']}"
        else:
            active_kind = str(state["active_side"])
            active_update = _update_active_candidate(active_kind, state, row, int(index), work, stats, config)
            feature_row[f"{active_kind}_break_attempt_on_close_flag"] = bool(active_update["new_break"])
            chosen_event = _peek_confirmation(active_kind, state, row, int(index), config)
            if chosen_event is not None:
                _apply_confirmation(state, chosen_event, row, int(index), stats, config)
                if active_kind == "high":
                    feature_row["confirmed_high_on_close_flag"] = True
                    feature_row["confirmed_high_label_on_close"] = chosen_event["structure_label"]
                    feature_row["confirmed_high_value_on_close"] = chosen_event["value"]
                    feature_row["confirmed_high_swing_date_on_close"] = chosen_event["swing_date"]
                    feature_row["confirmed_high_bars_to_confirmation_on_close"] = chosen_event[
                        "bars_to_confirmation"
                    ]
                else:
                    feature_row["confirmed_low_on_close_flag"] = True
                    feature_row["confirmed_low_label_on_close"] = chosen_event["structure_label"]
                    feature_row["confirmed_low_value_on_close"] = chosen_event["value"]
                    feature_row["confirmed_low_swing_date_on_close"] = chosen_event["swing_date"]
                    feature_row["confirmed_low_bars_to_confirmation_on_close"] = chosen_event[
                        "bars_to_confirmation"
                    ]
                feature_row["confirmation_event_on_close"] = f"confirmed_{active_kind}"

        state["market_bias"] = break_event["market_bias_after_close"]
        feature_rows.append(feature_row)

    features = pd.DataFrame(feature_rows)
    candidate_highs = pd.DataFrame(stats["candidate_high_events"])
    candidate_lows = pd.DataFrame(stats["candidate_low_events"])
    confirmed_highs = pd.DataFrame(stats["confirmed_high_events"])
    confirmed_lows = pd.DataFrame(stats["confirmed_low_events"])
    break_attempt_highs = pd.DataFrame(stats["break_attempt_high_events"])
    break_attempt_lows = pd.DataFrame(stats["break_attempt_low_events"])
    structure_breaks = pd.DataFrame(stats["structure_break_events"])

    summary = summarize_structure(
        config,
        ohlcv,
        features,
        confirmed_highs,
        confirmed_lows,
        structure_breaks,
        stats,
    )
    return StructureArtifacts(
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


def summarize_structure(
    config: BtcStructureConfig,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    confirmed_highs: pd.DataFrame,
    confirmed_lows: pd.DataFrame,
    structure_breaks: pd.DataFrame,
    stats: dict[str, Any],
) -> dict[str, Any]:
    high_label_counts = (
        confirmed_highs["structure_label"].value_counts(dropna=False).sort_index().to_dict()
        if not confirmed_highs.empty
        else {}
    )
    low_label_counts = (
        confirmed_lows["structure_label"].value_counts(dropna=False).sort_index().to_dict()
        if not confirmed_lows.empty
        else {}
    )
    break_counts = (
        structure_breaks["event"].value_counts(dropna=False).sort_index().to_dict()
        if not structure_breaks.empty
        else {}
    )
    latest_close = float(ohlcv["close"].iloc[-1])
    latest_feature = features.iloc[-1]
    return {
        "ticker": config.ticker,
        "market_type": config.market_type.value,
        "interval": config.interval,
        "years_requested": config.years,
        "bars": int(len(ohlcv)),
        "start": pd.Timestamp(ohlcv["date"].iloc[0]).isoformat(),
        "end": pd.Timestamp(ohlcv["date"].iloc[-1]).isoformat(),
        "latest_close": latest_close,
        "confirmed_highs": int(len(confirmed_highs)),
        "confirmed_lows": int(len(confirmed_lows)),
        "avg_bars_to_confirm_high": (
            float(np.mean(stats["bars_to_confirm_high"])) if stats["bars_to_confirm_high"] else None
        ),
        "avg_bars_to_confirm_low": (
            float(np.mean(stats["bars_to_confirm_low"])) if stats["bars_to_confirm_low"] else None
        ),
        "candidate_high_replaced_before_confirmation": int(
            stats["candidate_high_replaced_before_confirmation"]
        ),
        "candidate_low_replaced_before_confirmation": int(
            stats["candidate_low_replaced_before_confirmation"]
        ),
        "high_label_counts": {str(key): int(value) for key, value in high_label_counts.items()},
        "low_label_counts": {str(key): int(value) for key, value in low_label_counts.items()},
        "structure_break_counts": {str(key): int(value) for key, value in break_counts.items()},
        "last_market_bias_after_close": latest_feature["market_bias_after_close"],
        "last_confirmed_high_value_asof": _serialize_value(latest_feature["high_confirmed_value_asof"]),
        "last_confirmed_low_value_asof": _serialize_value(latest_feature["low_confirmed_value_asof"]),
        "config": {
            "rolling_lookback": config.rolling_lookback,
            "atr_window": config.atr_window,
            "atr_multiplier": config.atr_multiplier,
            "pct_threshold": config.pct_threshold,
            "min_bars_confirmation": config.min_bars_confirmation,
            "force_confirmation_after_bars": config.force_confirmation_after_bars,
            "max_candidate_bars": config.max_candidate_bars,
            "level_windows": list(config.level_windows),
            "level_confluence_required": config.level_confluence_required,
            "level_tolerance_atr_multiplier": config.level_tolerance_atr_multiplier,
            "require_multi_horizon_confluence": config.require_multi_horizon_confluence,
            "short_confluence_max_window": config.short_confluence_max_window,
            "long_confluence_min_window": config.long_confluence_min_window,
            "min_short_confluence_hits": config.min_short_confluence_hits,
            "min_long_confluence_hits": config.min_long_confluence_hits,
            "candidate_replace_min_atr_step": config.candidate_replace_min_atr_step,
            "candidate_replace_min_pct_step": config.candidate_replace_min_pct_step,
            "hhll_tolerance_atr_multiplier": config.hhll_tolerance_atr_multiplier,
            "hhll_tolerance_pct": config.hhll_tolerance_pct,
            "bos_choch_atr_multiplier": config.bos_choch_atr_multiplier,
            "bos_choch_pct": config.bos_choch_pct,
        },
    }


def _filter_candidate_events_for_plot(
    events_df: pd.DataFrame,
    *,
    confluence_required: int = 2,
    min_confluence_for_candidate: int = 3,
    min_gap_bars: int = 5,
    include_seeded_candidates: bool = False,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    out = events_df.copy().sort_values(["swing_date", "available_on"])
    seeded_or_reanchored = out["event_type"].astype(str).str.contains("seeded|reanchored", regex=True, na=False)
    structure_break = out["breaks_structure"].fillna(False)
    confluence_floor = max(int(confluence_required), int(min_confluence_for_candidate))
    strong_confluence = out["confluence_count"].fillna(0) >= confluence_floor
    non_replace = ~out["event_type"].astype(str).str.endswith("_replaced")
    seeded_gate = seeded_or_reanchored if include_seeded_candidates else pd.Series(False, index=out.index)
    out = out[seeded_gate | structure_break | (strong_confluence & non_replace)]
    out = out.drop_duplicates(subset=["swing_date", "value"], keep="last")
    if out.empty or int(min_gap_bars) <= 0:
        return out

    keep_idx: list[int] = []
    last_kept_index: int | None = None
    for idx, row in out.iterrows():
        current_index = int(row.get("confluence_count", 0))
        swing_ts = pd.Timestamp(row["swing_date"])
        if last_kept_index is None:
            keep_idx.append(idx)
            last_kept_index = int(row.name)
            last_kept_swing = swing_ts
            continue
        if len(keep_idx) == 0 or (swing_ts - last_kept_swing).days >= int(min_gap_bars):
            keep_idx.append(idx)
            last_kept_index = int(row.name)
            last_kept_swing = swing_ts
            continue
    return out.loc[keep_idx]


def _filter_break_attempt_events_for_plot(
    events_df: pd.DataFrame,
    *,
    min_confluence: int = 3,
    min_gap_bars: int = 8,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()
    out = events_df.copy().sort_values(["swing_date", "available_on"])
    out = out[out["confluence_count"].fillna(0) >= int(min_confluence)]
    if out.empty or int(min_gap_bars) <= 0:
        return out
    keep_idx: list[int] = []
    last_kept_swing: pd.Timestamp | None = None
    for idx, row in out.iterrows():
        swing_ts = pd.Timestamp(row["swing_date"])
        if last_kept_swing is None or (swing_ts - last_kept_swing).days >= int(min_gap_bars):
            keep_idx.append(idx)
            last_kept_swing = swing_ts
    return out.loc[keep_idx]


def _filter_confirmed_events_for_plot(
    events_df: pd.DataFrame,
    *,
    min_gap_bars: int = 0,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()
    out = events_df.copy().sort_values(["swing_date", "available_on"])
    out = out.drop_duplicates(subset=["swing_date", "value"], keep="last")
    if int(min_gap_bars) <= 0:
        return out
    keep_idx: list[int] = []
    last_kept_swing: pd.Timestamp | None = None
    for idx, row in out.iterrows():
        swing_ts = pd.Timestamp(row["swing_date"])
        if last_kept_swing is None or (swing_ts - last_kept_swing).days >= int(min_gap_bars):
            keep_idx.append(idx)
            last_kept_swing = swing_ts
    return out.loc[keep_idx]


def build_structure_figure(
    artifacts: StructureArtifacts,
    *,
    title: str | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    include_candidates: bool = True,
    include_break_attempts: bool = True,
    include_structure_breaks: bool = True,
) -> Any | None:
    if go is None:
        return None

    frame = artifacts.ohlcv.copy()
    if start_date is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("selected plot window has no BTC rows")

    start_ts = pd.Timestamp(frame["date"].iloc[0])
    end_ts = pd.Timestamp(frame["date"].iloc[-1])

    def _slice_events(events_df: pd.DataFrame, date_col: str = "available_on") -> pd.DataFrame:
        if events_df.empty:
            return events_df.copy()
        out = events_df.copy()
        out[date_col] = pd.to_datetime(out[date_col], utc=True)
        return out[(out[date_col] >= start_ts) & (out[date_col] <= end_ts)]

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="BTC",
        )
    )

    if include_candidates:
        candidate_highs = _filter_candidate_events_for_plot(_slice_events(artifacts.candidate_highs, "available_on"))
        candidate_lows = _filter_candidate_events_for_plot(_slice_events(artifacts.candidate_lows, "available_on"))
        if not candidate_highs.empty:
            figure.add_trace(
                go.Scatter(
                    x=candidate_highs["swing_date"],
                    y=candidate_highs["value"],
                    mode="markers",
                    name="Candidate High",
                    marker={"color": "orange", "symbol": "triangle-up", "size": 10},
                    opacity=0.55,
                    text=candidate_highs["confluence_windows"],
                    hovertemplate=(
                        "Candidate High<br>Date=%{x}<br>Value=%{y:.2f}<br>Confluence=%{text}<extra></extra>"
                    ),
                )
            )
        if not candidate_lows.empty:
            figure.add_trace(
                go.Scatter(
                    x=candidate_lows["swing_date"],
                    y=candidate_lows["value"],
                    mode="markers",
                    name="Candidate Low",
                    marker={"color": "cyan", "symbol": "triangle-down", "size": 10},
                    opacity=0.55,
                    text=candidate_lows["confluence_windows"],
                    hovertemplate=(
                        "Candidate Low<br>Date=%{x}<br>Value=%{y:.2f}<br>Confluence=%{text}<extra></extra>"
                    ),
                )
            )

    if include_break_attempts:
        break_highs = _filter_break_attempt_events_for_plot(_slice_events(artifacts.break_attempt_highs))
        break_lows = _filter_break_attempt_events_for_plot(_slice_events(artifacts.break_attempt_lows))
        if not break_highs.empty:
            figure.add_trace(
                go.Scatter(
                    x=break_highs["swing_date"],
                    y=break_highs["value"],
                    mode="markers",
                    name="Fresh Break-Up Attempt",
                    marker={"color": "darkorange", "symbol": "star", "size": 13},
                    text=break_highs["confluence_windows"],
                    hovertemplate=(
                        "Break-Up Attempt<br>Date=%{x}<br>Value=%{y:.2f}<br>Confluence=%{text}<extra></extra>"
                    ),
                )
            )
        if not break_lows.empty:
            figure.add_trace(
                go.Scatter(
                    x=break_lows["swing_date"],
                    y=break_lows["value"],
                    mode="markers",
                    name="Fresh Break-Down Attempt",
                    marker={"color": "deepskyblue", "symbol": "star", "size": 13},
                    text=break_lows["confluence_windows"],
                    hovertemplate=(
                        "Break-Down Attempt<br>Date=%{x}<br>Value=%{y:.2f}<br>Confluence=%{text}<extra></extra>"
                    ),
                )
            )

    confirmed_highs = _filter_confirmed_events_for_plot(_slice_events(artifacts.confirmed_highs))
    confirmed_lows = _filter_confirmed_events_for_plot(_slice_events(artifacts.confirmed_lows))
    if not confirmed_highs.empty:
        figure.add_trace(
            go.Scatter(
                x=confirmed_highs["swing_date"],
                y=confirmed_highs["value"],
                mode="markers+text",
                marker={"color": "#d62728", "size": 11, "symbol": "triangle-down"},
                text=confirmed_highs["structure_label"],
                textposition="top center",
                name="Confirmed Highs",
            )
        )
    if not confirmed_lows.empty:
        figure.add_trace(
            go.Scatter(
                x=confirmed_lows["swing_date"],
                y=confirmed_lows["value"],
                mode="markers+text",
                marker={"color": "#2ca02c", "size": 11, "symbol": "triangle-up"},
                text=confirmed_lows["structure_label"],
                textposition="bottom center",
                name="Confirmed Lows",
            )
        )
    if include_structure_breaks and not artifacts.structure_breaks.empty:
        structure_breaks = _slice_events(artifacts.structure_breaks)
        color_map = {
            "choch_up": "#1f77b4",
            "choch_down": "#9467bd",
            "bos_up": "#17becf",
            "bos_down": "#ff7f0e",
        }
        if not structure_breaks.empty:
            figure.add_trace(
                go.Scatter(
                    x=structure_breaks["available_on"],
                    y=structure_breaks["close"],
                    mode="markers+text",
                    marker={
                        "size": 9,
                        "color": [
                            color_map.get(str(event), "#7f7f7f")
                            for event in structure_breaks["event"]
                        ],
                        "symbol": "diamond",
                    },
                    text=structure_breaks["event"].str.upper(),
                    textposition="middle right",
                    name="BOS / CHoCH",
                )
            )

    figure.update_layout(
        title=title or f"{artifacts.summary['ticker']} causal structure ({artifacts.summary['interval']})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=720,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return figure


def build_major_structure_figure(
    artifacts: StructureArtifacts,
    *,
    title: str | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
) -> Any | None:
    ranked_highs = rank_confirmed_levels(artifacts.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(artifacts.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(
        artifacts.structure_breaks,
        artifacts.confirmed_highs,
        artifacts.confirmed_lows,
    )

    filtered = StructureArtifacts(
        ohlcv=artifacts.ohlcv,
        features=artifacts.features,
        candidate_highs=artifacts.candidate_highs.iloc[0:0].copy(),
        candidate_lows=artifacts.candidate_lows.iloc[0:0].copy(),
        confirmed_highs=ranked_highs[ranked_highs["is_strategy_level"]].copy(),
        confirmed_lows=ranked_lows[ranked_lows["is_strategy_level"]].copy(),
        break_attempt_highs=artifacts.break_attempt_highs.iloc[0:0].copy(),
        break_attempt_lows=artifacts.break_attempt_lows.iloc[0:0].copy(),
        structure_breaks=ranked_breaks[ranked_breaks["is_strategy_break"]].copy(),
        summary=artifacts.summary,
    )
    return build_structure_figure(
        filtered,
        title=title or f"{artifacts.summary['ticker']} major structure only ({artifacts.summary['interval']})",
        start_date=start_date,
        end_date=end_date,
        include_candidates=False,
        include_break_attempts=False,
        include_structure_breaks=True,
    )


def build_global_major_structure_figure(
    artifacts: StructureArtifacts,
    *,
    title: str | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
) -> Any | None:
    ranked_highs = rank_confirmed_levels(artifacts.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(artifacts.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(
        artifacts.structure_breaks,
        artifacts.confirmed_highs,
        artifacts.confirmed_lows,
    )
    filtered = StructureArtifacts(
        ohlcv=artifacts.ohlcv,
        features=artifacts.features,
        candidate_highs=artifacts.candidate_highs.iloc[0:0].copy(),
        candidate_lows=artifacts.candidate_lows.iloc[0:0].copy(),
        confirmed_highs=ranked_highs[ranked_highs["level_scope"].isin(["major", "global"])].copy(),
        confirmed_lows=ranked_lows[ranked_lows["level_scope"].isin(["major", "global"])].copy(),
        break_attempt_highs=artifacts.break_attempt_highs.iloc[0:0].copy(),
        break_attempt_lows=artifacts.break_attempt_lows.iloc[0:0].copy(),
        structure_breaks=ranked_breaks[ranked_breaks["broken_level_scope"].isin(["major", "global"])].copy(),
        summary=artifacts.summary,
    )
    return build_structure_figure(
        filtered,
        title=title or f"{artifacts.summary['ticker']} global/major structure only ({artifacts.summary['interval']})",
        start_date=start_date,
        end_date=end_date,
        include_candidates=False,
        include_break_attempts=False,
        include_structure_breaks=True,
    )


def plot_structure_last_n_bars(
    artifacts: StructureArtifacts,
    *,
    last_n: int = 320,
    title: str | None = None,
) -> Any | None:
    frame = artifacts.ohlcv.tail(int(last_n))
    if frame.empty:
        return None
    return build_structure_figure(
        artifacts,
        start_date=frame["date"].iloc[0],
        end_date=frame["date"].iloc[-1],
        title=title or f"Static structure view - last {int(last_n)} bars",
    )


def save_structure_artifacts(
    artifacts: StructureArtifacts,
    output_dir: Path | str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts.ohlcv.to_csv(output_path / "btc_ohlcv.csv", index=False)
    artifacts.features.to_csv(output_path / "btc_structure_features.csv", index=False)
    artifacts.candidate_highs.to_csv(output_path / "candidate_highs.csv", index=False)
    artifacts.candidate_lows.to_csv(output_path / "candidate_lows.csv", index=False)
    artifacts.confirmed_highs.to_csv(output_path / "confirmed_highs.csv", index=False)
    artifacts.confirmed_lows.to_csv(output_path / "confirmed_lows.csv", index=False)
    artifacts.break_attempt_highs.to_csv(output_path / "break_attempt_highs.csv", index=False)
    artifacts.break_attempt_lows.to_csv(output_path / "break_attempt_lows.csv", index=False)
    artifacts.structure_breaks.to_csv(output_path / "structure_breaks.csv", index=False)
    rank_confirmed_levels(artifacts.confirmed_highs, kind="high").to_csv(
        output_path / "confirmed_highs_ranked.csv",
        index=False,
    )
    rank_confirmed_levels(artifacts.confirmed_lows, kind="low").to_csv(
        output_path / "confirmed_lows_ranked.csv",
        index=False,
    )
    rank_structure_breaks(
        artifacts.structure_breaks,
        artifacts.confirmed_highs,
        artifacts.confirmed_lows,
    ).to_csv(output_path / "structure_breaks_ranked.csv", index=False)

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.summary, handle, indent=2, ensure_ascii=False)

    figure = build_structure_figure(artifacts)
    if figure is not None:
        figure.write_html(output_path / "btc_structure_chart.html")
        recent_figure = plot_structure_last_n_bars(artifacts, last_n=min(320, len(artifacts.ohlcv)))
        if recent_figure is not None:
            recent_figure.write_html(output_path / "btc_structure_last_320_bars.html")
        major_figure = build_major_structure_figure(artifacts)
        if major_figure is not None:
            major_figure.write_html(output_path / "btc_structure_major_only.html")
            recent_major_figure = build_major_structure_figure(
                artifacts,
                start_date=artifacts.ohlcv["date"].iloc[max(0, len(artifacts.ohlcv) - min(320, len(artifacts.ohlcv)))],
                end_date=artifacts.ohlcv["date"].iloc[-1],
                title=f"{artifacts.summary['ticker']} major structure - last {min(320, len(artifacts.ohlcv))} bars",
            )
            if recent_major_figure is not None:
                recent_major_figure.write_html(output_path / "btc_structure_major_only_last_320_bars.html")
        strict_figure = build_global_major_structure_figure(artifacts)
        if strict_figure is not None:
            strict_figure.write_html(output_path / "btc_structure_global_major_only.html")
            recent_strict_figure = build_global_major_structure_figure(
                artifacts,
                start_date=artifacts.ohlcv["date"].iloc[max(0, len(artifacts.ohlcv) - min(320, len(artifacts.ohlcv)))],
                end_date=artifacts.ohlcv["date"].iloc[-1],
                title=f"{artifacts.summary['ticker']} global/major structure - last {min(320, len(artifacts.ohlcv))} bars",
            )
            if recent_strict_figure is not None:
                recent_strict_figure.write_html(output_path / "btc_structure_global_major_only_last_320_bars.html")

    return output_path


def run_btc_structure_pipeline(
    config: BtcStructureConfig,
    *,
    output_dir: Path | str | None = None,
    end: datetime | None = None,
    client: BinanceClient | None = None,
) -> StructureArtifacts:
    ohlcv = fetch_btc_ohlcv(config, end=end, client=client)
    artifacts = simulate_btc_structure(ohlcv, config)
    if output_dir is not None:
        save_structure_artifacts(artifacts, output_dir)
    return artifacts
