from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


BASE_LEVEL_DAYS = (1, 3, 7, 10, 30, 90, 180, 300, 365)
FIB_BASE_RATIOS = (0.0, 0.236, 0.34, 0.382, 0.5, 0.618, 0.66, 0.786, 1.0)
FIB_EXTENSION_RATIOS = (1.272, 1.618, 2.0)


@dataclass(slots=True, frozen=True)
class BtcStructureConfig:
    ticker: str = "BTC/USDT"
    market_type: str = "futures"
    interval: str = "1d"
    years: int = 5

    rolling_lookback: int = 400
    atr_window: int = 14
    atr_multiplier: float = 1.25
    pct_threshold: float = 0.015
    min_bars_confirmation: int = 3
    force_confirmation_after_bars: int = 7
    max_candidate_bars: int = 18

    level_windows: tuple[int, ...] = BASE_LEVEL_DAYS
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
        market_type: str = "futures",
        years: int = 5,
    ) -> "BtcStructureConfig":
        seconds = interval_to_seconds(interval)
        bars_per_day = max(1, int(round(86400 / seconds)))

        def bars(days: int) -> int:
            return max(1, int(round(days * bars_per_day)))

        return BtcStructureConfig(
            ticker=ticker,
            market_type=market_type.lower(),
            interval=interval,
            years=years,
            rolling_lookback=max(bars(400), bars(365) + bars(30)),
            atr_window=bars(14),
            min_bars_confirmation=max(2, bars(3)),
            force_confirmation_after_bars=max(bars(7), bars(3) + 2),
            max_candidate_bars=max(bars(18), bars(7) + bars(3)),
            level_windows=tuple(sorted({bars(days) for days in BASE_LEVEL_DAYS})),
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


@dataclass(slots=True)
class StructureLabArtifacts:
    ranked_highs: pd.DataFrame
    ranked_lows: pd.DataFrame
    ranked_breaks: pd.DataFrame
    feature_matrix: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class StructureExperimentResult:
    config: BtcStructureConfig
    structure: StructureArtifacts
    lab: StructureLabArtifacts


def interval_to_seconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    raise ValueError(f"unsupported interval format: {interval}")


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.upper().replace("/", "").replace("-", "")
    if not normalized.endswith("USDT"):
        raise ValueError("this standalone file currently expects USDT symbols like BTC/USDT")
    return normalized


def _binance_kline_url(market_type: str) -> str:
    normalized = market_type.strip().lower()
    if normalized == "futures":
        return "https://fapi.binance.com/fapi/v1/klines"
    if normalized == "spot":
        return "https://api.binance.com/api/v3/klines"
    raise ValueError(f"unsupported market_type: {market_type}")


def _fetch_klines_rest(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    *,
    market_type: str,
    limit: int = 1000,
) -> list[list[Any]]:
    endpoint = _binance_kline_url(market_type)
    rows: list[list[Any]] = []
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    interval_ms = interval_to_seconds(interval) * 1000
    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        url = f"{endpoint}?{urlencode(params)}"
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload:
            break
        rows.extend(payload)
        last_open = int(payload[-1][0])
        next_start_ms = last_open + interval_ms
        if next_start_ms <= start_ms:
            break
        start_ms = next_start_ms
        if len(payload) < limit:
            break
    return rows


def fetch_btc_ohlcv(config: BtcStructureConfig, *, end: datetime | None = None) -> pd.DataFrame:
    end_dt = end or datetime.now(UTC)
    start_dt = end_dt - timedelta(days=365 * config.years)
    rows = _fetch_klines_rest(
        _normalize_symbol(config.ticker),
        config.interval,
        start_dt,
        end_dt,
        market_type=config.market_type,
    )
    data = [
        {
            "date": datetime.fromtimestamp(int(item[6]) / 1000, tz=UTC),
            "ticker": config.ticker,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        }
        for item in rows
    ]
    frame = pd.DataFrame(data)
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
        out[f"rolling_high_{window}"] = out["high"].shift(1).rolling(window=window, min_periods=min_periods).max()
        out[f"rolling_low_{window}"] = out["low"].shift(1).rolling(window=window, min_periods=min_periods).min()
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
    passed, mode_used, metrics, threshold = _confirmation_signal(kind, row, candidate, bars_since_candidate, config)
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
        int(candidate["confluence_count"]) >= config.level_confluence_required and multi_horizon_ok
    ) or bool(candidate["breaks_structure"])
    if not confluence_ok and bars_since_candidate < config.force_confirmation_after_bars:
        return None
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
        "confirmation_path": "confluence" if confluence_ok else "forced_maturity",
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
) -> None:
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
    if bool(seeded["breaks_structure"]):
        _push_break_attempt_event(stats, other, seeded, row["date"])


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
    bos_up = bos_down = choch_up = choch_down = False
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
            choch_up = bias_asof == "bearish"
            bos_up = not choch_up
            bias_after_close = "bullish"
    if confirmed_low is not None:
        down_threshold = _break_threshold(float(confirmed_low), atr_value, config)
        if float(row["close"]) < float(confirmed_low) - down_threshold:
            if broken_level_kind is None:
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


def summarize_structure(
    config: BtcStructureConfig,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    confirmed_highs: pd.DataFrame,
    confirmed_lows: pd.DataFrame,
    structure_breaks: pd.DataFrame,
    stats: dict[str, Any],
) -> dict[str, Any]:
    high_counts = confirmed_highs["structure_label"].value_counts(dropna=False).sort_index().to_dict() if not confirmed_highs.empty else {}
    low_counts = confirmed_lows["structure_label"].value_counts(dropna=False).sort_index().to_dict() if not confirmed_lows.empty else {}
    break_counts = structure_breaks["event"].value_counts(dropna=False).sort_index().to_dict() if not structure_breaks.empty else {}
    latest_feature = features.iloc[-1]
    return {
        "ticker": config.ticker,
        "market_type": config.market_type,
        "interval": config.interval,
        "years_requested": config.years,
        "bars": int(len(ohlcv)),
        "start": pd.Timestamp(ohlcv["date"].iloc[0]).isoformat(),
        "end": pd.Timestamp(ohlcv["date"].iloc[-1]).isoformat(),
        "latest_close": float(ohlcv["close"].iloc[-1]),
        "confirmed_highs": int(len(confirmed_highs)),
        "confirmed_lows": int(len(confirmed_lows)),
        "avg_bars_to_confirm_high": float(np.mean(stats["bars_to_confirm_high"])) if stats["bars_to_confirm_high"] else None,
        "avg_bars_to_confirm_low": float(np.mean(stats["bars_to_confirm_low"])) if stats["bars_to_confirm_low"] else None,
        "candidate_high_replaced_before_confirmation": int(stats["candidate_high_replaced_before_confirmation"]),
        "candidate_low_replaced_before_confirmation": int(stats["candidate_low_replaced_before_confirmation"]),
        "high_label_counts": {str(k): int(v) for k, v in high_counts.items()},
        "low_label_counts": {str(k): int(v) for k, v in low_counts.items()},
        "structure_break_counts": {str(k): int(v) for k, v in break_counts.items()},
        "last_market_bias_after_close": latest_feature["market_bias_after_close"],
        "last_confirmed_high_value_asof": _serialize_value(latest_feature["high_confirmed_value_asof"]),
        "last_confirmed_low_value_asof": _serialize_value(latest_feature["low_confirmed_value_asof"]),
    }


def simulate_btc_structure(ohlcv: pd.DataFrame, config: BtcStructureConfig) -> StructureArtifacts:
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

        feature_row = {
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
                else:
                    feature_row["confirmed_low_on_close_flag"] = True
                    feature_row["confirmed_low_label_on_close"] = chosen_event["structure_label"]
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
                else:
                    feature_row["confirmed_low_on_close_flag"] = True
                    feature_row["confirmed_low_label_on_close"] = chosen_event["structure_label"]

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
    summary = summarize_structure(config, ohlcv, features, confirmed_highs, confirmed_lows, structure_breaks, stats)
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


def _parse_confluence_windows(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    out: list[int] = []
    for item in [part.strip() for part in str(value).split(",") if part.strip()]:
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def fib_ratio_label(ratio: float) -> str:
    text = f"{float(ratio):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _mirror_ratio(ratio: float) -> float:
    return 1.0 - float(ratio)


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


def _classify_swing_tier(max_window: int, confluence_count: int, confluence_long_hits: int) -> str:
    if int(confluence_count) >= 4 and int(confluence_long_hits) >= 2 and int(max_window) >= 180:
        return "global_extrema"
    if int(confluence_count) >= 3 and int(confluence_long_hits) >= 1 and int(max_window) >= 30:
        return "structural_extrema"
    return "local_extrema"


def _tier_bonus(tier: str) -> float:
    normalized = str(tier)
    if normalized == "global_extrema":
        return 1.25
    if normalized == "structural_extrema":
        return 0.75
    return 0.25


def rank_confirmed_levels(confirmed_df: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    if confirmed_df.empty:
        return confirmed_df.copy()
    ranked = confirmed_df.copy()
    ranked["available_on"] = pd.to_datetime(ranked["available_on"], utc=True)
    ranked["swing_date"] = pd.to_datetime(ranked["swing_date"], utc=True)
    ranked["confluence_window_list"] = ranked["confluence_windows"].apply(_parse_confluence_windows)
    ranked["max_window"] = ranked["confluence_window_list"].apply(lambda items: max(items) if items else 0)
    ranked["level_scope"] = ranked["max_window"].apply(_classify_level_scope)
    ranked["window_score"] = ranked["max_window"].apply(_window_score)
    ranked["structure_bonus"] = ranked["breaks_structure"].fillna(False).astype(int) * 2
    ranked["label_bonus"] = ranked["structure_label"].map(
        {"HH": 2 if kind == "high" else 0, "LL": 2 if kind == "low" else 0, "HL": 1 if kind == "low" else 0, "LH": 1 if kind == "high" else 0, "EQH": 1 if kind == "high" else 0, "EQL": 1 if kind == "low" else 0}
    ).fillna(0)
    ranked["level_score"] = ranked["confluence_count"].fillna(0).astype(int) + ranked["window_score"] + ranked["structure_bonus"] + ranked["label_bonus"]
    ranked["swing_tier"] = [
        _classify_swing_tier(max_window, confluence_count, confluence_long_hits)
        for max_window, confluence_count, confluence_long_hits in zip(
            ranked["max_window"].fillna(0),
            ranked["confluence_count"].fillna(0),
            ranked["confluence_long_hits"].fillna(0),
            strict=False,
        )
    ]
    ranked["level_priority"] = pd.cut(ranked["level_score"], bins=[-1, 5, 8, 11, 100], labels=["low", "medium", "high", "critical"]).astype(str)
    ranked["is_strategy_level"] = ranked["level_scope"].isin(["structural", "major", "global"]) & (ranked["level_score"] >= 7)
    return ranked


def rank_structure_breaks(structure_breaks: pd.DataFrame, confirmed_highs: pd.DataFrame, confirmed_lows: pd.DataFrame) -> pd.DataFrame:
    if structure_breaks.empty:
        return structure_breaks.copy()
    ranked = structure_breaks.copy()
    ranked["available_on"] = pd.to_datetime(ranked["available_on"], utc=True)
    ranked_highs = rank_confirmed_levels(confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(confirmed_lows, kind="low")
    scopes: list[str] = []
    scores: list[float] = []
    priorities: list[str] = []
    for _, row in ranked.iterrows():
        source = ranked_highs if row.get("level_kind") == "high" else ranked_lows
        value = row.get("level_value")
        when = pd.Timestamp(row["available_on"])
        if source.empty or pd.isna(value):
            scopes.append("unknown"); scores.append(np.nan); priorities.append("unranked"); continue
        matches = source[np.isclose(source["value"].astype(float), float(value), rtol=0.0, atol=1e-9) & (source["available_on"] <= when)].sort_values("available_on")
        if matches.empty:
            scopes.append("unknown"); scores.append(np.nan); priorities.append("unranked"); continue
        match = matches.iloc[-1]
        scopes.append(str(match["level_scope"]))
        scores.append(float(match["level_score"]))
        priorities.append(str(match["level_priority"]))
    ranked["broken_level_scope"] = scopes
    ranked["broken_level_score"] = scores
    ranked["broken_level_priority"] = priorities
    ranked["is_major_break"] = ranked["broken_level_scope"].isin(["major", "global"])
    ranked["is_strategy_break"] = ranked["event"].astype(str).str.startswith("choch") | (ranked["event"].astype(str).str.startswith("bos") & ranked["is_major_break"])
    return ranked


def filter_ranked_levels(ranked_df: pd.DataFrame, *, scopes: tuple[str, ...] | None = None, min_level_score: float | None = None, priorities: tuple[str, ...] | None = None, labels: tuple[str, ...] | None = None, only_strategy_levels: bool = False) -> pd.DataFrame:
    if ranked_df.empty:
        return ranked_df.copy()
    out = ranked_df.copy()
    if scopes is not None:
        out = out[out["level_scope"].isin(scopes)]
    if min_level_score is not None:
        out = out[out["level_score"] >= float(min_level_score)]
    if priorities is not None:
        out = out[out["level_priority"].isin(priorities)]
    if labels is not None:
        out = out[out["structure_label"].isin(labels)]
    if only_strategy_levels:
        out = out[out["is_strategy_level"].fillna(False)]
    return out.sort_values(["available_on", "swing_date"]).reset_index(drop=True)


def filter_ranked_breaks(ranked_df: pd.DataFrame, *, scopes: tuple[str, ...] | None = None, events: tuple[str, ...] | None = None, min_level_score: float | None = None, priorities: tuple[str, ...] | None = None, only_strategy_breaks: bool = False) -> pd.DataFrame:
    if ranked_df.empty:
        return ranked_df.copy()
    out = ranked_df.copy()
    if scopes is not None:
        out = out[out["broken_level_scope"].isin(scopes)]
    if events is not None:
        out = out[out["event"].isin(events)]
    if min_level_score is not None:
        out = out[out["broken_level_score"] >= float(min_level_score)]
    if priorities is not None:
        out = out[out["broken_level_priority"].isin(priorities)]
    if only_strategy_breaks:
        out = out[out["is_strategy_break"].fillna(False)]
    return out.sort_values("available_on").reset_index(drop=True)


def _merge_last_levels(base: pd.DataFrame, ranked_df: pd.DataFrame, *, kind: str, scope: str) -> pd.DataFrame:
    side = "high" if kind == "high" else "low"
    prefix = f"{scope}_{side}"
    part = ranked_df[ranked_df["level_scope"] == scope].copy()
    if part.empty:
        base[f"{prefix}_value"] = np.nan
        base[f"{prefix}_score"] = np.nan
        base[f"{prefix}_label"] = None
        base[f"{prefix}_available_on"] = pd.NaT
        base[f"{prefix}_days_since"] = np.nan
        return base
    part = part.sort_values("available_on")[["available_on", "swing_date", "value", "structure_label", "level_score"]].rename(
        columns={
            "available_on": f"{prefix}_available_on",
            "swing_date": f"{prefix}_swing_date",
            "value": f"{prefix}_value",
            "structure_label": f"{prefix}_label",
            "level_score": f"{prefix}_score",
        }
    )
    merged = pd.merge_asof(base.sort_values("date"), part, left_on="date", right_on=f"{prefix}_available_on", direction="backward")
    delta = merged["date"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _merge_last_breaks(base: pd.DataFrame, ranked_breaks: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    prefix = f"{scope}_break"
    part = ranked_breaks[ranked_breaks["broken_level_scope"] == scope].copy()
    if part.empty:
        base[f"{prefix}_event"] = None
        base[f"{prefix}_available_on"] = pd.NaT
        base[f"{prefix}_score"] = np.nan
        base[f"{prefix}_days_since"] = np.nan
        return base
    part = part.sort_values("available_on")[["available_on", "event", "broken_level_score"]].rename(
        columns={
            "available_on": f"{prefix}_available_on",
            "event": f"{prefix}_event",
            "broken_level_score": f"{prefix}_score",
        }
    )
    merged = pd.merge_asof(base.sort_values("date"), part, left_on="date", right_on=f"{prefix}_available_on", direction="backward")
    delta = merged["date"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _attach_distance_features(base: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    high_col = f"{scope}_high_value"
    low_col = f"{scope}_low_value"
    base[f"{scope}_distance_to_high_pct"] = np.where(base[high_col].notna(), base["close"] / base[high_col] - 1.0, np.nan)
    base[f"{scope}_distance_to_low_pct"] = np.where(base[low_col].notna(), base["close"] / base[low_col] - 1.0, np.nan)
    width = base[high_col] - base[low_col]
    base[f"{scope}_box_position"] = np.where(width.abs() > 1e-9, (base["close"] - base[low_col]) / width, np.nan)
    return base


def _rolling_event_counts(base: pd.DataFrame, ranked_breaks: pd.DataFrame, *, scopes: tuple[str, ...], windows: tuple[int, ...] = (7, 30, 90)) -> pd.DataFrame:
    if ranked_breaks.empty:
        return base
    daily = base[["date"]].copy()
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    breaks = ranked_breaks.copy()
    breaks["available_on"] = pd.to_datetime(breaks["available_on"], utc=True)
    event_counts = (
        breaks.assign(event_key=breaks["broken_level_scope"].astype(str) + "__" + breaks["event"].astype(str))
        .groupby(["available_on", "event_key"])
        .size()
        .unstack(fill_value=0)
    )
    merged = daily.merge(event_counts, how="left", left_on="date", right_index=True).fillna(0)
    for scope in scopes:
        for window in windows:
            up_cols = [col for col in [f"{scope}__bos_up", f"{scope}__choch_up"] if col in merged.columns]
            down_cols = [col for col in [f"{scope}__bos_down", f"{scope}__choch_down"] if col in merged.columns]
            merged[f"{scope}_up_events_{window}"] = merged[up_cols].sum(axis=1).rolling(window=window, min_periods=1).sum() if up_cols else 0.0
            merged[f"{scope}_down_events_{window}"] = merged[down_cols].sum(axis=1).rolling(window=window, min_periods=1).sum() if down_cols else 0.0
            merged[f"{scope}_net_break_pressure_{window}"] = merged[f"{scope}_up_events_{window}"] - merged[f"{scope}_down_events_{window}"]
    keep_cols = ["date"] + [col for col in merged.columns if col != "date" and "__" not in col]
    return base.merge(merged[keep_cols], on="date", how="left")


def _fib_scope_settings(scope: str) -> dict[str, Any]:
    normalized = str(scope)
    if normalized == "local":
        return {
            "source_scopes": ("local", "structural"),
            "min_level_score": 3.0,
            "lookback_days": 150,
            "top_n": 6,
            "recency_penalty": 0.060,
        }
    if normalized == "structural":
        return {
            "source_scopes": ("structural",),
            "min_level_score": 5.0,
            "lookback_days": 260,
            "top_n": 8,
            "recency_penalty": 0.035,
        }
    if normalized == "major":
        return {
            "source_scopes": ("structural", "major"),
            "min_level_score": 6.0,
            "lookback_days": 640,
            "top_n": 10,
            "recency_penalty": 0.018,
        }
    if normalized == "global":
        return {
            "source_scopes": ("global",),
            "min_level_score": 7.0,
            "lookback_days": None,
            "top_n": 12,
            "recency_penalty": 0.006,
        }
    raise ValueError(f"unsupported fib scope: {scope}")


def _filter_fib_source_levels(ranked_df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    settings = _fib_scope_settings(scope)
    out = ranked_df[ranked_df["level_scope"].isin(settings["source_scopes"])].copy()
    out = out[out["level_score"] >= float(settings["min_level_score"])]
    return out.sort_values("available_on").reset_index(drop=True)


def _candidate_pool_for_fib(levels: pd.DataFrame, current_ts: pd.Timestamp, *, scope: str) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()
    settings = _fib_scope_settings(scope)
    out = levels[levels["available_on"] <= current_ts].copy()
    lookback_days = settings["lookback_days"]
    if lookback_days is not None:
        cutoff = current_ts - pd.Timedelta(days=int(lookback_days))
        out = out[out["available_on"] >= cutoff]
    if out.empty:
        return out
    top_n = int(settings["top_n"])
    recent = out.sort_values("available_on").tail(top_n)
    strongest = out.sort_values(["level_score", "available_on"], ascending=[False, False]).head(top_n)
    pooled = pd.concat([recent, strongest], ignore_index=True)
    pooled = pooled.drop_duplicates(subset=["available_on", "swing_date", "value"], keep="first")
    return pooled.sort_values("available_on").reset_index(drop=True)


def _score_fib_pair(high_row: pd.Series, low_row: pd.Series, current_ts: pd.Timestamp, *, scope: str) -> tuple[float, str] | None:
    high_value = float(high_row["value"])
    low_value = float(low_row["value"])
    if not np.isfinite(high_value) or not np.isfinite(low_value) or high_value <= low_value:
        return None

    high_swing = pd.Timestamp(high_row["swing_date"])
    low_swing = pd.Timestamp(low_row["swing_date"])
    if high_swing == low_swing:
        return None
    leg_direction = "bullish" if high_swing > low_swing else "bearish"

    high_available = pd.Timestamp(high_row["available_on"])
    low_available = pd.Timestamp(low_row["available_on"])
    newer_available = max(high_available, low_available)
    older_available = min(high_available, low_available)
    newer_age_days = max(0.0, (current_ts - newer_available).total_seconds() / 86400.0)
    older_age_days = max(0.0, (current_ts - older_available).total_seconds() / 86400.0)
    swing_span_days = abs((high_swing - low_swing).total_seconds() / 86400.0)

    settings = _fib_scope_settings(scope)
    score = float(high_row["level_score"]) + float(low_row["level_score"])
    score += _tier_bonus(high_row.get("swing_tier", "")) + _tier_bonus(low_row.get("swing_tier", ""))
    if leg_direction == "bullish" and str(high_row.get("structure_label", "")) in {"HH", "EQH"}:
        score += 0.75
    if leg_direction == "bearish" and str(low_row.get("structure_label", "")) in {"LL", "EQL"}:
        score += 0.75
    score -= newer_age_days * float(settings["recency_penalty"])
    score -= older_age_days * float(settings["recency_penalty"]) * 0.20

    lookback_days = settings["lookback_days"]
    if lookback_days is not None and swing_span_days > float(lookback_days):
        score -= (swing_span_days - float(lookback_days)) * float(settings["recency_penalty"]) * 0.10

    return score, leg_direction


def _select_active_fib_pair(
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    current_ts: pd.Timestamp,
    *,
    scope: str,
) -> dict[str, Any] | None:
    high_pool = _candidate_pool_for_fib(highs, current_ts, scope=scope)
    low_pool = _candidate_pool_for_fib(lows, current_ts, scope=scope)
    if high_pool.empty or low_pool.empty:
        return None

    best: dict[str, Any] | None = None
    for _, high_row in high_pool.iterrows():
        for _, low_row in low_pool.iterrows():
            scored = _score_fib_pair(high_row, low_row, current_ts, scope=scope)
            if scored is None:
                continue
            pair_score, leg_direction = scored
            candidate = {
                "pair_score_adjusted": float(pair_score),
                "leg_direction": leg_direction,
                "anchor_high_date": pd.Timestamp(high_row["swing_date"]),
                "anchor_high": float(high_row["value"]),
                "high_score": float(high_row["level_score"]),
                "high_label": high_row.get("structure_label"),
                "high_tier": high_row.get("swing_tier"),
                "anchor_low_date": pd.Timestamp(low_row["swing_date"]),
                "anchor_low": float(low_row["value"]),
                "low_score": float(low_row["level_score"]),
                "low_label": low_row.get("structure_label"),
                "low_tier": low_row.get("swing_tier"),
                "pair_key": f"{pd.Timestamp(low_row['swing_date']).date()}|{float(low_row['value']):.8f}__{pd.Timestamp(high_row['swing_date']).date()}|{float(high_row['value']):.8f}",
                "pair_available_on": max(pd.Timestamp(high_row["available_on"]), pd.Timestamp(low_row["available_on"])),
            }
            if best is None:
                best = candidate
                continue
            if candidate["pair_score_adjusted"] > best["pair_score_adjusted"]:
                best = candidate
                continue
            if candidate["pair_score_adjusted"] == best["pair_score_adjusted"] and candidate["pair_available_on"] > best["pair_available_on"]:
                best = candidate
    return best


def _build_fib_leg_features(
    base: pd.DataFrame,
    ranked_highs: pd.DataFrame,
    ranked_lows: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    prefix = f"{scope}_fib"
    highs = _filter_fib_source_levels(ranked_highs, scope=scope)
    lows = _filter_fib_source_levels(ranked_lows, scope=scope)
    if highs.empty or lows.empty:
        for col in [
            f"{prefix}_anchor_low",
            f"{prefix}_anchor_high",
            f"{prefix}_range",
            f"{prefix}_pair_score",
            f"{prefix}_leg_position",
        ]:
            base[col] = np.nan
        for col in [
            f"{prefix}_anchor_low_date",
            f"{prefix}_anchor_high_date",
            f"{prefix}_leg_direction",
            f"{prefix}_pair_key",
        ]:
            base[col] = None
        base[f"{prefix}_anchor_changed_flag"] = False
        return base

    work = base.sort_values("date").reset_index(drop=True).copy()
    pair_rows: list[dict[str, Any]] = []
    last_pair: dict[str, Any] | None = None
    for current_ts in pd.to_datetime(work["date"], utc=True):
        selected = _select_active_fib_pair(highs, lows, current_ts, scope=scope)
        if selected is not None:
            last_pair = selected
        pair_rows.append(last_pair.copy() if last_pair is not None else {})

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        pair_df = pd.DataFrame(index=work.index)

    for col, default in [
        ("pair_score_adjusted", np.nan),
        ("leg_direction", None),
        ("anchor_high_date", pd.NaT),
        ("anchor_high", np.nan),
        ("high_score", np.nan),
        ("high_label", None),
        ("high_tier", None),
        ("anchor_low_date", pd.NaT),
        ("anchor_low", np.nan),
        ("low_score", np.nan),
        ("low_label", None),
        ("low_tier", None),
        ("pair_key", None),
        ("pair_available_on", pd.NaT),
    ]:
        if col not in pair_df.columns:
            pair_df[col] = default

    pair_df["range"] = pair_df["anchor_high"] - pair_df["anchor_low"]
    active_pair = pair_df["pair_key"].notna() & pair_df["range"].gt(0)

    work[f"{prefix}_high_available_on"] = pd.to_datetime(pair_df["pair_available_on"], utc=True, errors="coerce")
    work[f"{prefix}_anchor_high_date"] = pd.to_datetime(pair_df["anchor_high_date"], utc=True, errors="coerce")
    work[f"{prefix}_anchor_high"] = pair_df["anchor_high"]
    work[f"{prefix}_high_score"] = pair_df["high_score"]
    work[f"{prefix}_high_label"] = pair_df["high_label"]
    work[f"{prefix}_high_tier"] = pair_df["high_tier"]
    work[f"{prefix}_low_available_on"] = pd.to_datetime(pair_df["pair_available_on"], utc=True, errors="coerce")
    work[f"{prefix}_anchor_low_date"] = pd.to_datetime(pair_df["anchor_low_date"], utc=True, errors="coerce")
    work[f"{prefix}_anchor_low"] = pair_df["anchor_low"]
    work[f"{prefix}_low_score"] = pair_df["low_score"]
    work[f"{prefix}_low_label"] = pair_df["low_label"]
    work[f"{prefix}_low_tier"] = pair_df["low_tier"]
    work[f"{prefix}_range"] = pair_df["range"].where(active_pair)
    work[f"{prefix}_leg_direction"] = pair_df["leg_direction"].where(active_pair, None)
    work[f"{prefix}_pair_score"] = pair_df["pair_score_adjusted"].where(active_pair)
    work[f"{prefix}_pair_key"] = pair_df["pair_key"].where(active_pair, None)
    work[f"{prefix}_anchor_changed_flag"] = work[f"{prefix}_pair_key"].ne(work[f"{prefix}_pair_key"].shift(1)) & active_pair
    work[f"{prefix}_direction_changed_flag"] = work[f"{prefix}_leg_direction"].ne(work[f"{prefix}_leg_direction"].shift(1)) & active_pair
    work[f"{prefix}_leg_position"] = np.where(
        active_pair,
        (work["close"] - work[f"{prefix}_anchor_low"]) / work[f"{prefix}_range"],
        np.nan,
    )

    for ratio in FIB_BASE_RATIOS:
        label = fib_ratio_label(ratio)
        col = f"{prefix}_{label}_level"
        work[col] = np.where(
            active_pair,
            work[f"{prefix}_anchor_low"] + work[f"{prefix}_range"] * float(ratio),
            np.nan,
        )
        work[f"{prefix}_distance_to_{label}_pct"] = np.where(
            work[col].notna(),
            work["close"] / work[col] - 1.0,
            np.nan,
        )
        mirror_label = fib_ratio_label(_mirror_ratio(ratio))
        mirror_col = f"{prefix}_mirror_{label}_level"
        work[mirror_col] = np.where(
            active_pair,
            work[f"{prefix}_anchor_low"] + work[f"{prefix}_range"] * _mirror_ratio(ratio),
            np.nan,
        )
        work[f"{prefix}_directional_{label}_level"] = np.where(
            active_pair & work[f"{prefix}_leg_direction"].eq("bullish"),
            work[col],
            np.where(
                active_pair & work[f"{prefix}_leg_direction"].eq("bearish"),
                work[mirror_col],
                np.nan,
            ),
        )
        work[f"{prefix}_directional_distance_to_{label}_pct"] = np.where(
            work[f"{prefix}_directional_{label}_level"].notna(),
            work["close"] / work[f"{prefix}_directional_{label}_level"] - 1.0,
            np.nan,
        )
        work[f"{prefix}_directional_alias_{label}"] = np.where(
            active_pair & work[f"{prefix}_leg_direction"].eq("bullish"),
            label,
            np.where(active_pair & work[f"{prefix}_leg_direction"].eq("bearish"), mirror_label, None),
        )

    for ratio in FIB_EXTENSION_RATIOS:
        label = fib_ratio_label(ratio)
        col = f"{prefix}_ext_{label}_level"
        up_col = f"{prefix}_ext_up_{label}_level"
        down_col = f"{prefix}_ext_down_{label}_level"
        work[up_col] = np.where(
            active_pair,
            work[f"{prefix}_anchor_high"] + work[f"{prefix}_range"] * (float(ratio) - 1.0),
            np.nan,
        )
        work[down_col] = np.where(
            active_pair,
            work[f"{prefix}_anchor_low"] - work[f"{prefix}_range"] * (float(ratio) - 1.0),
            np.nan,
        )
        work[col] = np.where(
            active_pair & work[f"{prefix}_leg_direction"].eq("bullish"),
            work[up_col],
            np.where(
                active_pair & work[f"{prefix}_leg_direction"].eq("bearish"),
                work[down_col],
                np.nan,
            ),
        )
        work[f"{prefix}_distance_to_ext_{label}_pct"] = np.where(
            work[col].notna(),
            work["close"] / work[col] - 1.0,
            np.nan,
        )
        work[f"{prefix}_distance_to_ext_up_{label}_pct"] = np.where(
            work[up_col].notna(),
            work["close"] / work[up_col] - 1.0,
            np.nan,
        )
        work[f"{prefix}_distance_to_ext_down_{label}_pct"] = np.where(
            work[down_col].notna(),
            work["close"] / work[down_col] - 1.0,
            np.nan,
        )

    return work


def _merge_last_break_event(base: pd.DataFrame, ranked_breaks: pd.DataFrame, *, scope: str, event_name: str) -> pd.DataFrame:
    prefix = f"{scope}_{event_name}"
    part = ranked_breaks[
        (ranked_breaks["broken_level_scope"] == scope) & (ranked_breaks["event"] == event_name)
    ].copy()
    if part.empty:
        base[f"{prefix}_available_on"] = pd.NaT
        base[f"{prefix}_score"] = np.nan
        base[f"{prefix}_days_since"] = np.nan
        return base
    part = part.sort_values("available_on")[["available_on", "broken_level_score"]].rename(
        columns={
            "available_on": f"{prefix}_available_on",
            "broken_level_score": f"{prefix}_score",
        }
    )
    merged = pd.merge_asof(base.sort_values("date"), part, left_on="date", right_on=f"{prefix}_available_on", direction="backward")
    delta = merged["date"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _between_series(value: pd.Series, left: pd.Series, right: pd.Series) -> pd.Series:
    low = pd.concat([left, right], axis=1).min(axis=1)
    high = pd.concat([left, right], axis=1).max(axis=1)
    return value.notna() & low.notna() & high.notna() & value.ge(low) & value.le(high)


def _attach_scope_break_state_features(base: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    event_col = f"{scope}_break_event"
    base[f"{scope}_last_break_is_bullish"] = base[event_col].isin(["bos_up", "choch_up"])
    base[f"{scope}_last_break_is_bearish"] = base[event_col].isin(["bos_down", "choch_down"])
    base[f"{scope}_last_break_is_choch_up"] = base[event_col].eq("choch_up")
    base[f"{scope}_last_break_is_choch_down"] = base[event_col].eq("choch_down")
    base[f"{scope}_last_break_is_bos_up"] = base[event_col].eq("bos_up")
    base[f"{scope}_last_break_is_bos_down"] = base[event_col].eq("bos_down")
    return base


def _attach_scope_fib_zone_features(base: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    prefix = f"{scope}_fib"
    close = base["close"]
    midpoint = base[f"{prefix}_0_5_level"]
    lower_pocket_left = base[f"{prefix}_0_34_level"]
    lower_pocket_right = base[f"{prefix}_0_382_level"]
    upper_pocket_left = base[f"{prefix}_0_618_level"]
    upper_pocket_right = base[f"{prefix}_0_66_level"]
    ext_1618 = base[f"{prefix}_ext_1_618_level"]
    leg_direction = base[f"{prefix}_leg_direction"]

    near_threshold = np.maximum(base["atr"].fillna(0.0) * 0.50, close.abs() * 0.003)

    base[f"{prefix}_leg_is_bullish"] = leg_direction.eq("bullish")
    base[f"{prefix}_leg_is_bearish"] = leg_direction.eq("bearish")
    base[f"{prefix}_above_midpoint"] = midpoint.notna() & close.ge(midpoint)
    base[f"{prefix}_below_midpoint"] = midpoint.notna() & close.le(midpoint)
    base[f"{prefix}_in_lower_pocket"] = _between_series(close, lower_pocket_left, lower_pocket_right)
    base[f"{prefix}_in_upper_pocket"] = _between_series(close, upper_pocket_left, upper_pocket_right)
    base[f"{prefix}_in_full_premium_zone"] = _between_series(close, midpoint, base[f"{prefix}_1_level"])
    base[f"{prefix}_in_full_discount_zone"] = _between_series(close, base[f"{prefix}_0_level"], midpoint)
    base[f"{prefix}_near_anchor_low"] = base[f"{prefix}_anchor_low"].notna() & (close - base[f"{prefix}_anchor_low"]).abs().le(near_threshold)
    base[f"{prefix}_near_anchor_high"] = base[f"{prefix}_anchor_high"].notna() & (close - base[f"{prefix}_anchor_high"]).abs().le(near_threshold)
    base[f"{prefix}_near_ext_1_618"] = ext_1618.notna() & (close - ext_1618).abs().le(near_threshold)
    base[f"{prefix}_bullish_support_zone_flag"] = base[f"{prefix}_leg_is_bullish"] & base[f"{prefix}_in_upper_pocket"]
    base[f"{prefix}_bearish_resistance_zone_flag"] = base[f"{prefix}_leg_is_bearish"] & base[f"{prefix}_in_lower_pocket"]
    base[f"{prefix}_directional_0_618_0_66_zone_flag"] = _between_series(
        close,
        base[f"{prefix}_directional_0_618_level"],
        base[f"{prefix}_directional_0_66_level"],
    )
    base[f"{prefix}_directional_0_34_0_382_zone_flag"] = _between_series(
        close,
        base[f"{prefix}_directional_0_34_level"],
        base[f"{prefix}_directional_0_382_level"],
    )
    base[f"{prefix}_leg_flipped_recently_30d_flag"] = (
        base[f"{prefix}_direction_changed_flag"].fillna(False).rolling(window=30, min_periods=1).max().astype(bool)
    )
    return base


def _attach_cross_scope_structure_features(base: pd.DataFrame) -> pd.DataFrame:
    base["major_global_break_alignment_bullish"] = base["major_last_break_is_bullish"] & base["global_last_break_is_bullish"]
    base["major_global_break_alignment_bearish"] = base["major_last_break_is_bearish"] & base["global_last_break_is_bearish"]
    base["major_global_fib_alignment_bullish"] = base["major_fib_leg_is_bullish"] & base["global_fib_leg_is_bullish"]
    base["major_global_fib_alignment_bearish"] = base["major_fib_leg_is_bearish"] & base["global_fib_leg_is_bearish"]
    base["major_global_bullish_confluence_flag"] = base["major_global_break_alignment_bullish"] & base["major_global_fib_alignment_bullish"]
    base["major_global_bearish_confluence_flag"] = base["major_global_break_alignment_bearish"] & base["major_global_fib_alignment_bearish"]
    base["major_global_break_disagreement_flag"] = base["major_last_break_is_bullish"] ^ base["global_last_break_is_bullish"]
    base["major_global_fib_disagreement_flag"] = base["major_fib_leg_is_bullish"] ^ base["global_fib_leg_is_bullish"]

    base["major_pullback_long_candidate_flag"] = (
        base["major_global_bullish_confluence_flag"]
        & _between_series(base["close"], base["major_fib_0_5_level"], base["major_fib_0_66_level"])
        & base["global_fib_above_midpoint"]
    )
    base["major_pullback_short_candidate_flag"] = (
        base["major_global_bearish_confluence_flag"]
        & _between_series(base["close"], base["major_fib_0_34_level"], base["major_fib_0_5_level"])
        & base["global_fib_below_midpoint"]
    )
    base["global_continuation_long_flag"] = (
        base["global_last_break_is_bullish"]
        & base["global_fib_leg_is_bullish"]
        & base["global_fib_above_midpoint"]
    )
    base["global_continuation_short_flag"] = (
        base["global_last_break_is_bearish"]
        & base["global_fib_leg_is_bearish"]
        & base["global_fib_below_midpoint"]
    )
    return base


def build_structure_feature_matrix(structure: StructureArtifacts) -> pd.DataFrame:
    ranked_highs = rank_confirmed_levels(structure.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(structure.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(structure.structure_breaks, structure.confirmed_highs, structure.confirmed_lows)
    base = structure.features[["date", "ticker", "open", "high", "low", "close", "volume", "atr", "market_bias_asof", "market_bias_after_close", "structure_break_event_on_close", "choch_any_on_close_flag", "bos_any_on_close_flag"]].copy()
    base["date"] = pd.to_datetime(base["date"], utc=True)
    scopes = ("local", "structural", "major", "global")
    for scope in scopes:
        base = _merge_last_levels(base, ranked_highs, kind="high", scope=scope)
        base = _merge_last_levels(base, ranked_lows, kind="low", scope=scope)
        base = _merge_last_breaks(base, ranked_breaks, scope=scope)
        base = _attach_distance_features(base, scope=scope)
    base = _rolling_event_counts(base, ranked_breaks, scopes=scopes)
    base = _build_fib_leg_features(base, ranked_highs, ranked_lows, scope="local")
    base = _build_fib_leg_features(base, ranked_highs, ranked_lows, scope="major")
    base = _build_fib_leg_features(base, ranked_highs, ranked_lows, scope="global")
    base = base.copy()
    for scope in scopes:
        base = _attach_scope_break_state_features(base, scope=scope)
        for event_name in ("bos_up", "bos_down", "choch_up", "choch_down"):
            base = _merge_last_break_event(base, ranked_breaks, scope=scope, event_name=event_name)
    base = base.copy()
    for scope in ("local", "major", "global"):
        base = _attach_scope_fib_zone_features(base, scope=scope)
    base = base.copy()
    base = _attach_cross_scope_structure_features(base)
    return base.sort_values("date").reset_index(drop=True)


def summarize_structure_lab(lab: StructureLabArtifacts) -> dict[str, Any]:
    breaks = lab.ranked_breaks
    return {
        "ranked_highs": int(len(lab.ranked_highs)),
        "ranked_lows": int(len(lab.ranked_lows)),
        "ranked_breaks": int(len(breaks)),
        "break_scope_counts": {str(k): int(v) for k, v in breaks["broken_level_scope"].value_counts().to_dict().items()} if not breaks.empty else {},
        "strategy_break_counts": {str(k): int(v) for k, v in breaks[breaks["is_strategy_break"].fillna(False)]["event"].value_counts().to_dict().items()} if not breaks.empty else {},
        "feature_rows": int(len(lab.feature_matrix)),
        "feature_columns": list(lab.feature_matrix.columns),
    }


def run_structure_feature_lab(structure: StructureArtifacts) -> StructureLabArtifacts:
    ranked_highs = rank_confirmed_levels(structure.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(structure.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(structure.structure_breaks, structure.confirmed_highs, structure.confirmed_lows)
    feature_matrix = build_structure_feature_matrix(structure)
    lab = StructureLabArtifacts(ranked_highs=ranked_highs, ranked_lows=ranked_lows, ranked_breaks=ranked_breaks, feature_matrix=feature_matrix, summary={})
    lab.summary = summarize_structure_lab(lab)
    return lab


def run_btc_structure_lab(*, interval: str = "1d", ticker: str = "BTC/USDT", market_type: str = "futures", years: int = 5) -> StructureExperimentResult:
    config = BtcStructureConfig.for_interval(interval, ticker=ticker, market_type=market_type, years=years)
    structure = simulate_btc_structure(fetch_btc_ohlcv(config), config)
    lab = run_structure_feature_lab(structure)
    return StructureExperimentResult(config=config, structure=structure, lab=lab)


def summarize_structure_experiment(result: StructureExperimentResult) -> dict[str, Any]:
    ranked_breaks = result.lab.ranked_breaks
    fm = result.lab.feature_matrix
    last_row = fm.iloc[-1]
    return {
        "ticker": result.config.ticker,
        "interval": result.config.interval,
        "bars": int(len(result.structure.ohlcv)),
        "confirmed_highs": int(len(result.structure.confirmed_highs)),
        "confirmed_lows": int(len(result.structure.confirmed_lows)),
        "ranked_breaks": int(len(ranked_breaks)),
        "strategy_breaks": int(ranked_breaks["is_strategy_break"].fillna(False).sum()),
        "major_global_breaks": int(ranked_breaks["broken_level_scope"].isin(["major", "global"]).sum()),
        "feature_matrix_shape": tuple(result.lab.feature_matrix.shape),
        "last_bias": result.structure.summary.get("last_market_bias_after_close"),
        "local_fib_leg_direction": last_row.get("local_fib_leg_direction"),
        "major_fib_leg_direction": last_row.get("major_fib_leg_direction"),
        "global_fib_leg_direction": last_row.get("global_fib_leg_direction"),
        "local_fib_anchor_changed_flag": bool(last_row.get("local_fib_anchor_changed_flag", False)),
        "major_fib_anchor_changed_flag": bool(last_row.get("major_fib_anchor_changed_flag", False)),
        "global_fib_anchor_changed_flag": bool(last_row.get("global_fib_anchor_changed_flag", False)),
        "local_lower_pocket_hits": int(fm["local_fib_in_lower_pocket"].fillna(False).sum()) if "local_fib_in_lower_pocket" in fm.columns else 0,
        "local_upper_pocket_hits": int(fm["local_fib_in_upper_pocket"].fillna(False).sum()) if "local_fib_in_upper_pocket" in fm.columns else 0,
        "major_lower_pocket_hits": int(fm["major_fib_in_lower_pocket"].fillna(False).sum()) if "major_fib_in_lower_pocket" in fm.columns else 0,
        "major_upper_pocket_hits": int(fm["major_fib_in_upper_pocket"].fillna(False).sum()) if "major_fib_in_upper_pocket" in fm.columns else 0,
        "global_lower_pocket_hits": int(fm["global_fib_in_lower_pocket"].fillna(False).sum()) if "global_fib_in_lower_pocket" in fm.columns else 0,
        "global_upper_pocket_hits": int(fm["global_fib_in_upper_pocket"].fillna(False).sum()) if "global_fib_in_upper_pocket" in fm.columns else 0,
        "major_pullback_long_candidates": int(fm["major_pullback_long_candidate_flag"].fillna(False).sum()) if "major_pullback_long_candidate_flag" in fm.columns else 0,
        "major_pullback_short_candidates": int(fm["major_pullback_short_candidate_flag"].fillna(False).sum()) if "major_pullback_short_candidate_flag" in fm.columns else 0,
        "global_continuation_long_flags": int(fm["global_continuation_long_flag"].fillna(False).sum()) if "global_continuation_long_flag" in fm.columns else 0,
        "global_continuation_short_flags": int(fm["global_continuation_short_flag"].fillna(False).sum()) if "global_continuation_short_flag" in fm.columns else 0,
    }


def _slice_plot_frame(frame: pd.DataFrame, *, last_n: int | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    if last_n is not None and int(last_n) > 0:
        out = out.tail(int(last_n))
    return out


def _slice_events(events_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, *, date_col: str = "available_on") -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()
    out = events_df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=True)
    if "swing_date" in out.columns:
        out["swing_date"] = pd.to_datetime(out["swing_date"], utc=True)
    return out[(out[date_col] >= start_ts) & (out[date_col] <= end_ts)]


def _prepare_plot_datasets(
    result: StructureExperimentResult,
    *,
    mode: str,
    last_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _slice_plot_frame(result.structure.ohlcv, last_n=last_n)
    start_ts = pd.Timestamp(frame["date"].iloc[0])
    end_ts = pd.Timestamp(frame["date"].iloc[-1])

    if mode == "raw":
        highs = _slice_events(result.structure.confirmed_highs, start_ts, end_ts)
        lows = _slice_events(result.structure.confirmed_lows, start_ts, end_ts)
        breaks = _slice_events(result.structure.structure_breaks, start_ts, end_ts)
        candidates = pd.concat(
            [
                _slice_events(result.structure.candidate_highs, start_ts, end_ts),
                _slice_events(result.structure.candidate_lows, start_ts, end_ts),
            ],
            ignore_index=True,
        )
        return frame, highs, lows, breaks, candidates

    ranked_highs = rank_confirmed_levels(result.structure.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(result.structure.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(
        result.structure.structure_breaks,
        result.structure.confirmed_highs,
        result.structure.confirmed_lows,
    )

    if mode == "major":
        highs = filter_ranked_levels(ranked_highs, only_strategy_levels=True)
        lows = filter_ranked_levels(ranked_lows, only_strategy_levels=True)
        breaks = filter_ranked_breaks(ranked_breaks, only_strategy_breaks=True)
    elif mode == "local_global":
        highs = filter_ranked_levels(ranked_highs, scopes=("local", "global"))
        lows = filter_ranked_levels(ranked_lows, scopes=("local", "global"))
        breaks = filter_ranked_breaks(ranked_breaks, scopes=("local", "global"))
    elif mode == "global_major":
        highs = filter_ranked_levels(ranked_highs, scopes=("major", "global"))
        lows = filter_ranked_levels(ranked_lows, scopes=("major", "global"))
        breaks = filter_ranked_breaks(ranked_breaks, scopes=("major", "global"))
    else:
        raise ValueError(f"unsupported plot mode: {mode}")

    highs = _slice_events(highs, start_ts, end_ts)
    lows = _slice_events(lows, start_ts, end_ts)
    breaks = _slice_events(breaks, start_ts, end_ts)
    return frame, highs, lows, breaks, pd.DataFrame()


def _slice_feature_frame(feature_matrix: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    out = feature_matrix.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out[(out["date"] >= start_ts) & (out["date"] <= end_ts)].sort_values("date").reset_index(drop=True)


def _add_fib_plot_traces(fig: Any, feature_frame: pd.DataFrame, *, scopes: tuple[str, ...]) -> None:
    fib_levels = ("0_34", "0_5", "0_618", "0_66")
    scope_colors = {
        "local": {
            "0_34": "#d9d9d9",
            "0_5": "#bdbdbd",
            "0_618": "#969696",
            "0_66": "#737373",
            "ext_1_618": "#525252",
        },
        "major": {
            "0_34": "#8ecae6",
            "0_5": "#219ebc",
            "0_618": "#023047",
            "0_66": "#ffb703",
            "ext_1_618": "#fb8500",
        },
        "global": {
            "0_34": "#cdb4db",
            "0_5": "#b5179e",
            "0_618": "#7209b7",
            "0_66": "#560bad",
            "ext_1_618": "#3a0ca3",
        },
    }
    dash_map = {"0_34": "dot", "0_5": "dash", "0_618": "solid", "0_66": "dashdot", "ext_up_1_618": "longdash", "ext_down_1_618": "longdashdot"}

    def _ext_display_name(value: str) -> str:
        return value.replace("ext_up_1_618", "Ext Up 1.618").replace("ext_down_1_618", "Ext Down 1.618")

    for scope in scopes:
        if scope not in scope_colors:
            continue
        for level in fib_levels:
            col = f"{scope}_fib_{level}_level"
            if col not in feature_frame.columns or feature_frame[col].notna().sum() == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=feature_frame["date"],
                    y=feature_frame[col],
                    mode="lines",
                    line={"color": scope_colors[scope][level], "width": 1.2, "dash": dash_map[level]},
                    opacity=0.65,
                    name=f"{scope.title()} Fib {level.replace('_', '.')}",
                    hovertemplate=f"{scope.title()} Fib {level.replace('_', '.')}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )
        ext_specs = (
            ("ext_up_1_618", f"{scope}_fib_ext_up_1_618_level", 0.40),
            ("ext_down_1_618", f"{scope}_fib_ext_down_1_618_level", 0.40),
        )
        for ext_key, ext_col, opacity in ext_specs:
            if ext_col not in feature_frame.columns or feature_frame[ext_col].notna().sum() == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=feature_frame["date"],
                    y=feature_frame[ext_col],
                    mode="lines",
                    line={"color": scope_colors[scope]["ext_1_618"], "width": 1.1, "dash": dash_map[ext_key]},
                    opacity=opacity,
                    name=f"{scope.title()} Fib {_ext_display_name(ext_key)}",
                    hovertemplate=f"{scope.title()} Fib {_ext_display_name(ext_key)}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
                )
            )


def _add_global_extrema_plot_traces(fig: Any, result: StructureExperimentResult, *, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    highs = result.lab.ranked_highs.copy()
    lows = result.lab.ranked_lows.copy()
    if not highs.empty:
        highs["available_on"] = pd.to_datetime(highs["available_on"], utc=True)
        highs["swing_date"] = pd.to_datetime(highs["swing_date"], utc=True)
        highs = highs[
            ((highs.get("swing_tier") == "global_extrema") | (highs.get("level_scope") == "global"))
            & (highs["swing_date"] >= start_ts)
            & (highs["swing_date"] <= end_ts)
        ]
    if not lows.empty:
        lows["available_on"] = pd.to_datetime(lows["available_on"], utc=True)
        lows["swing_date"] = pd.to_datetime(lows["swing_date"], utc=True)
        lows = lows[
            ((lows.get("swing_tier") == "global_extrema") | (lows.get("level_scope") == "global"))
            & (lows["swing_date"] >= start_ts)
            & (lows["swing_date"] <= end_ts)
        ]
    if not highs.empty:
        fig.add_trace(
            go.Scatter(
                x=highs["swing_date"],
                y=highs["value"],
                mode="markers+text",
                marker={"color": "#7b2cbf", "symbol": "star-triangle-down", "size": 14, "line": {"width": 1, "color": "#3c096c"}},
                text=highs["structure_label"].astype(str).radd("G-"),
                textposition="top center",
                name="Global Highs",
                hovertemplate="Global High<br>%{x}<br>%{y:.2f}<extra></extra>",
            )
        )
    if not lows.empty:
        fig.add_trace(
            go.Scatter(
                x=lows["swing_date"],
                y=lows["value"],
                mode="markers+text",
                marker={"color": "#2b9348", "symbol": "star-triangle-up", "size": 14, "line": {"width": 1, "color": "#081c15"}},
                text=lows["structure_label"].astype(str).radd("G-"),
                textposition="bottom center",
                name="Global Lows",
                hovertemplate="Global Low<br>%{x}<br>%{y:.2f}<extra></extra>",
            )
        )


def build_structure_plot(
    result: StructureExperimentResult,
    *,
    mode: str = "raw",
    last_n: int | None = 320,
    include_candidates: bool | None = None,
) -> Any:
    if go is None:
        raise RuntimeError("plotly is not installed; install plotly to use --show-plot")

    frame, highs, lows, breaks, candidates = _prepare_plot_datasets(result, mode=mode, last_n=last_n)
    show_candidates = (mode == "raw") if include_candidates is None else bool(include_candidates)
    feature_frame = _slice_feature_frame(result.lab.feature_matrix, pd.Timestamp(frame["date"].iloc[0]), pd.Timestamp(frame["date"].iloc[-1]))

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=result.config.ticker,
        )
    )

    if mode == "major":
        _add_fib_plot_traces(fig, feature_frame, scopes=("major",))
    elif mode == "local_global":
        _add_fib_plot_traces(fig, feature_frame, scopes=("local", "global"))
    elif mode == "global_major":
        _add_fib_plot_traces(fig, feature_frame, scopes=("major", "global"))
    if mode in {"major", "local_global", "global_major"}:
        _add_global_extrema_plot_traces(
            fig,
            result,
            start_ts=pd.Timestamp(frame["date"].iloc[0]),
            end_ts=pd.Timestamp(frame["date"].iloc[-1]),
        )

    if show_candidates and not candidates.empty:
        high_candidates = candidates[candidates["event_type"].astype(str).str.contains("high", na=False)]
        low_candidates = candidates[candidates["event_type"].astype(str).str.contains("low", na=False)]
        if not high_candidates.empty:
            fig.add_trace(
                go.Scatter(
                    x=high_candidates["swing_date"],
                    y=high_candidates["value"],
                    mode="markers",
                    name="Candidate High",
                    marker={"color": "orange", "symbol": "triangle-up", "size": 8},
                    opacity=0.45,
                )
            )
        if not low_candidates.empty:
            fig.add_trace(
                go.Scatter(
                    x=low_candidates["swing_date"],
                    y=low_candidates["value"],
                    mode="markers",
                    name="Candidate Low",
                    marker={"color": "cyan", "symbol": "triangle-down", "size": 8},
                    opacity=0.45,
                )
            )

    if not highs.empty:
        fig.add_trace(
            go.Scatter(
                x=highs["swing_date"],
                y=highs["value"],
                mode="markers+text",
                marker={"color": "#d62728", "symbol": "triangle-down", "size": 10},
                text=highs["structure_label"],
                textposition="top center",
                name="Confirmed Highs",
            )
        )
    if not lows.empty:
        fig.add_trace(
            go.Scatter(
                x=lows["swing_date"],
                y=lows["value"],
                mode="markers+text",
                marker={"color": "#2ca02c", "symbol": "triangle-up", "size": 10},
                text=lows["structure_label"],
                textposition="bottom center",
                name="Confirmed Lows",
            )
        )
    if not breaks.empty:
        color_map = {
            "choch_up": "#1f77b4",
            "choch_down": "#9467bd",
            "bos_up": "#17becf",
            "bos_down": "#ff7f0e",
        }
        fig.add_trace(
            go.Scatter(
                x=breaks["available_on"],
                y=breaks["close"],
                mode="markers+text",
                marker={
                    "color": [color_map.get(str(item), "#7f7f7f") for item in breaks["event"]],
                    "symbol": "diamond",
                    "size": 9,
                },
                text=breaks["event"].astype(str).str.upper(),
                textposition="middle right",
                name="BOS / CHoCH",
            )
        )

    title_suffix = f"last {len(frame)} bars" if last_n is not None and len(frame) else "full"
    fig.update_layout(
        title=f"{result.config.ticker} structure plot | mode={mode} | {title_suffix}",
        template="plotly_white",
        height=760,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


def show_structure_plot(
    result: StructureExperimentResult,
    *,
    mode: str = "raw",
    last_n: int | None = 320,
    include_candidates: bool | None = None,
) -> Any:
    fig = build_structure_plot(result, mode=mode, last_n=last_n, include_candidates=include_candidates)
    fig.show()
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone BTC structure engine")
    parser.add_argument("--ticker", default="BTC/USDT")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--market-type", default="futures", choices=("futures", "spot"))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--show-plot", action="store_true", help="Open an interactive plot instead of only printing the summary")
    parser.add_argument(
        "--plot-mode",
        default="raw",
        choices=("raw", "major", "local_global", "global_major"),
        help="raw = all detected events, major = major fib, local_global = fast local + slow global fib, global_major = major/global levels",
    )
    parser.add_argument(
        "--plot-last-n",
        type=int,
        default=0,
        help="How many trailing bars to show in the plot. Use 0 to show the full history.",
    )
    parser.add_argument("--no-candidates", action="store_true", help="Hide candidate markers in raw plot mode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_btc_structure_lab(interval=args.interval, ticker=args.ticker, market_type=args.market_type, years=args.years)
    print(json.dumps(summarize_structure_experiment(result), indent=2, ensure_ascii=False))
    print("high_label_counts:", result.structure.summary.get("high_label_counts"))
    print("low_label_counts:", result.structure.summary.get("low_label_counts"))
    print("structure_break_counts:", result.structure.summary.get("structure_break_counts"))
    cols = [
        "date",
        "close",
        "market_bias_after_close",
        "major_high_value",
        "major_low_value",
        "major_break_event",
        "major_break_days_since",
        "global_break_event",
        "global_net_break_pressure_30",
        "local_fib_leg_direction",
        "local_fib_0_34_level",
        "local_fib_0_5_level",
        "local_fib_0_618_level",
        "local_fib_0_66_level",
        "local_fib_ext_1_618_level",
        "major_fib_leg_direction",
        "major_fib_0_34_level",
        "major_fib_0_5_level",
        "major_fib_0_618_level",
        "major_fib_0_66_level",
        "major_fib_ext_1_618_level",
        "global_fib_leg_direction",
        "global_fib_0_34_level",
        "global_fib_0_5_level",
        "global_fib_0_618_level",
        "global_fib_0_66_level",
        "global_fib_ext_1_618_level",
        "major_last_break_is_bullish",
        "global_last_break_is_bearish",
        "local_fib_in_lower_pocket",
        "local_fib_in_upper_pocket",
        "major_fib_in_lower_pocket",
        "major_fib_in_upper_pocket",
        "global_fib_in_lower_pocket",
        "global_fib_in_upper_pocket",
        "major_global_bullish_confluence_flag",
        "major_global_bearish_confluence_flag",
        "major_pullback_long_candidate_flag",
        "major_pullback_short_candidate_flag",
        "global_continuation_long_flag",
        "global_continuation_short_flag",
    ]
    print(result.lab.feature_matrix[cols].tail(10).to_string(index=False))
    if args.show_plot:
        show_structure_plot(
            result,
            mode=args.plot_mode,
            last_n=(None if args.plot_last_n == 0 else args.plot_last_n),
            include_candidates=not args.no_candidates,
        )


if __name__ == "__main__":
    main()
