from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FIB_BASE_RATIOS, FIB_EXTENSION_RATIOS
from .engine import StructureArtifacts
from .ranking import rank_confirmed_levels, rank_structure_breaks


# ---------------------------------------------------------------------------
# Lab artifacts
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StructureLabArtifacts:
    ranked_highs: pd.DataFrame
    ranked_lows: pd.DataFrame
    ranked_breaks: pd.DataFrame
    feature_matrix: pd.DataFrame
    summary: dict[str, Any]


@dataclass(slots=True)
class StructureExperimentResult:
    structure: StructureArtifacts
    lab: StructureLabArtifacts


# ---------------------------------------------------------------------------
# Fibonacci helpers
# ---------------------------------------------------------------------------

def fib_ratio_label(ratio: float) -> str:
    text = f"{ratio:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def _mirror_ratio(ratio: float) -> float:
    return 1.0 - ratio


def _tier_bonus(tier: str) -> float:
    if tier == "global_extrema":
        return 1.25
    if tier == "structural_extrema":
        return 0.75
    return 0.25


# ---------------------------------------------------------------------------
# Merge helpers — all use close_time as the temporal key
# ---------------------------------------------------------------------------

def _merge_last_levels(
    base: pd.DataFrame,
    ranked_df: pd.DataFrame,
    *,
    kind: str,
    scope: str,
) -> pd.DataFrame:
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
    part = part.sort_values("available_on")[
        ["available_on", "swing_date", "value", "structure_label", "level_score"]
    ].rename(columns={
        "available_on": f"{prefix}_available_on",
        "swing_date": f"{prefix}_swing_date",
        "value": f"{prefix}_value",
        "structure_label": f"{prefix}_label",
        "level_score": f"{prefix}_score",
    })
    merged = pd.merge_asof(
        base.sort_values("close_time"), part,
        left_on="close_time", right_on=f"{prefix}_available_on",
        direction="backward",
    )
    delta = merged["close_time"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _merge_last_breaks(
    base: pd.DataFrame,
    ranked_breaks: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    prefix = f"{scope}_break"
    part = ranked_breaks[ranked_breaks["broken_level_scope"] == scope].copy()
    if part.empty:
        base[f"{prefix}_event"] = None
        base[f"{prefix}_available_on"] = pd.NaT
        base[f"{prefix}_score"] = np.nan
        base[f"{prefix}_days_since"] = np.nan
        return base
    part = part.sort_values("available_on")[
        ["available_on", "event", "broken_level_score"]
    ].rename(columns={
        "available_on": f"{prefix}_available_on",
        "event": f"{prefix}_event",
        "broken_level_score": f"{prefix}_score",
    })
    merged = pd.merge_asof(
        base.sort_values("close_time"), part,
        left_on="close_time", right_on=f"{prefix}_available_on",
        direction="backward",
    )
    delta = merged["close_time"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _merge_last_break_event(
    base: pd.DataFrame,
    ranked_breaks: pd.DataFrame,
    *,
    scope: str,
    event_name: str,
) -> pd.DataFrame:
    prefix = f"{scope}_{event_name}"
    part = ranked_breaks[
        (ranked_breaks["broken_level_scope"] == scope) & (ranked_breaks["event"] == event_name)
    ].copy()
    if part.empty:
        base[f"{prefix}_available_on"] = pd.NaT
        base[f"{prefix}_score"] = np.nan
        base[f"{prefix}_days_since"] = np.nan
        return base
    part = part.sort_values("available_on")[
        ["available_on", "broken_level_score"]
    ].rename(columns={
        "available_on": f"{prefix}_available_on",
        "broken_level_score": f"{prefix}_score",
    })
    merged = pd.merge_asof(
        base.sort_values("close_time"), part,
        left_on="close_time", right_on=f"{prefix}_available_on",
        direction="backward",
    )
    delta = merged["close_time"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _attach_distance_features(base: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    high_col = f"{scope}_high_value"
    low_col = f"{scope}_low_value"
    base[f"{scope}_distance_to_high_pct"] = np.where(
        base[high_col].notna(), base["close"] / base[high_col] - 1.0, np.nan,
    )
    base[f"{scope}_distance_to_low_pct"] = np.where(
        base[low_col].notna(), base["close"] / base[low_col] - 1.0, np.nan,
    )
    width = base[high_col] - base[low_col]
    base[f"{scope}_box_position"] = np.where(
        width.abs() > 1e-9, (base["close"] - base[low_col]) / width, np.nan,
    )
    return base


def _rolling_event_counts(
    base: pd.DataFrame,
    ranked_breaks: pd.DataFrame,
    *,
    scopes: tuple[str, ...],
    windows: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    if ranked_breaks.empty:
        return base
    daily = base[["close_time"]].copy()
    daily["close_time"] = pd.to_datetime(daily["close_time"], utc=True)
    breaks = ranked_breaks.copy()
    breaks["available_on"] = pd.to_datetime(breaks["available_on"], utc=True)
    event_counts = (
        breaks.assign(event_key=breaks["broken_level_scope"].astype(str) + "__" + breaks["event"].astype(str))
        .groupby(["available_on", "event_key"])
        .size()
        .unstack(fill_value=0)
    )
    merged = daily.merge(event_counts, how="left", left_on="close_time", right_index=True).fillna(0)
    for scope in scopes:
        for window in windows:
            up_cols = [c for c in [f"{scope}__bos_up", f"{scope}__choch_up"] if c in merged.columns]
            down_cols = [c for c in [f"{scope}__bos_down", f"{scope}__choch_down"] if c in merged.columns]
            merged[f"{scope}_up_events_{window}"] = (
                merged[up_cols].sum(axis=1).rolling(window=window, min_periods=1).sum() if up_cols else 0.0
            )
            merged[f"{scope}_down_events_{window}"] = (
                merged[down_cols].sum(axis=1).rolling(window=window, min_periods=1).sum() if down_cols else 0.0
            )
            merged[f"{scope}_net_break_pressure_{window}"] = (
                merged[f"{scope}_up_events_{window}"] - merged[f"{scope}_down_events_{window}"]
            )
    keep_cols = ["close_time"] + [c for c in merged.columns if c != "close_time" and "__" not in c]
    return base.merge(merged[keep_cols], on="close_time", how="left")


# ---------------------------------------------------------------------------
# Fibonacci pair selection & level construction
# ---------------------------------------------------------------------------

def _fib_scope_settings(scope: str) -> dict[str, Any]:
    if scope == "local":
        return {"source_scopes": ("local", "structural"), "min_level_score": 3.0, "lookback_days": 150, "top_n": 6, "recency_penalty": 0.060}
    if scope == "structural":
        return {"source_scopes": ("structural",), "min_level_score": 5.0, "lookback_days": 260, "top_n": 8, "recency_penalty": 0.035}
    if scope == "major":
        return {"source_scopes": ("structural", "major"), "min_level_score": 6.0, "lookback_days": 640, "top_n": 10, "recency_penalty": 0.018}
    if scope == "global":
        return {"source_scopes": ("global",), "min_level_score": 7.0, "lookback_days": None, "top_n": 12, "recency_penalty": 0.006}
    raise ValueError(f"unsupported fib scope: {scope}")


def _filter_fib_source_levels(ranked_df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    settings = _fib_scope_settings(scope)
    out = ranked_df[ranked_df["level_scope"].isin(settings["source_scopes"])].copy()
    out = out[out["level_score"] >= float(settings["min_level_score"])]
    return out.sort_values("available_on").reset_index(drop=True)


def _candidate_pool_for_fib(
    levels: pd.DataFrame,
    current_ts: pd.Timestamp,
    *,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if levels.empty:
        return levels.copy()
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
    ordered_idx = list(recent.index)
    seen_idx = set(ordered_idx)
    for idx in strongest.index:
        if idx not in seen_idx:
            ordered_idx.append(idx)
            seen_idx.add(idx)
    return levels.loc[ordered_idx].sort_values("available_on").reset_index(drop=True)


def _score_fib_pair(
    high_row: Any,
    low_row: Any,
    current_ts: pd.Timestamp,
    *,
    settings: dict[str, Any],
) -> tuple[float, str] | None:
    high_value = float(high_row.value)
    low_value = float(low_row.value)
    if not np.isfinite(high_value) or not np.isfinite(low_value) or high_value <= low_value:
        return None

    high_swing = pd.Timestamp(high_row.swing_date)
    low_swing = pd.Timestamp(low_row.swing_date)
    if high_swing == low_swing:
        return None
    leg_direction = "bullish" if high_swing > low_swing else "bearish"

    high_available = pd.Timestamp(high_row.available_on)
    low_available = pd.Timestamp(low_row.available_on)
    newer_available = max(high_available, low_available)
    older_available = min(high_available, low_available)
    newer_age_days = max(0.0, (current_ts - newer_available).total_seconds() / 86400.0)
    older_age_days = max(0.0, (current_ts - older_available).total_seconds() / 86400.0)
    swing_span_days = abs((high_swing - low_swing).total_seconds() / 86400.0)

    score = float(high_row.level_score) + float(low_row.level_score)
    score += _tier_bonus(getattr(high_row, "swing_tier", "")) + _tier_bonus(getattr(low_row, "swing_tier", ""))
    if leg_direction == "bullish" and str(getattr(high_row, "structure_label", "")) in {"HH", "EQH"}:
        score += 0.75
    if leg_direction == "bearish" and str(getattr(low_row, "structure_label", "")) in {"LL", "EQL"}:
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
    settings = _fib_scope_settings(scope)
    high_pool = _candidate_pool_for_fib(highs, current_ts, settings=settings)
    low_pool = _candidate_pool_for_fib(lows, current_ts, settings=settings)
    if high_pool.empty or low_pool.empty:
        return None

    best: dict[str, Any] | None = None
    for high_row in high_pool.itertuples(index=False):
        for low_row in low_pool.itertuples(index=False):
            scored = _score_fib_pair(high_row, low_row, current_ts, settings=settings)
            if scored is None:
                continue
            pair_score, leg_direction = scored
            candidate = {
                "pair_score_adjusted": pair_score,
                "leg_direction": leg_direction,
                "anchor_high_date": pd.Timestamp(high_row.swing_date),
                "anchor_high": float(high_row.value),
                "high_score": float(high_row.level_score),
                "high_label": getattr(high_row, "structure_label", None),
                "high_tier": getattr(high_row, "swing_tier", None),
                "anchor_low_date": pd.Timestamp(low_row.swing_date),
                "anchor_low": float(low_row.value),
                "low_score": float(low_row.level_score),
                "low_label": getattr(low_row, "structure_label", None),
                "low_tier": getattr(low_row, "swing_tier", None),
                "pair_key": (
                    f"{pd.Timestamp(low_row.swing_date).date()}|{float(low_row.value):.8f}"
                    f"__{pd.Timestamp(high_row.swing_date).date()}|{float(high_row.value):.8f}"
                ),
                "pair_available_on": max(
                    pd.Timestamp(high_row.available_on),
                    pd.Timestamp(low_row.available_on),
                ),
            }
            if best is None or candidate["pair_score_adjusted"] > best["pair_score_adjusted"]:
                best = candidate
            elif candidate["pair_score_adjusted"] == best["pair_score_adjusted"] and candidate["pair_available_on"] > best["pair_available_on"]:
                best = candidate
    return best


# ---------------------------------------------------------------------------
# Optimized pair selection — replaces per-timestamp pandas sorts with
# pre-sorted arrays + bisect and caches the winning pair when the
# candidate pool is unchanged between consecutive timestamps.
# ---------------------------------------------------------------------------

_LevelRow = namedtuple("_LevelRow", [
    "value", "swing_date_ns", "available_on_ns",
    "level_score", "swing_tier", "structure_label",
    "swing_date_ts", "available_on_ts",
])

_NANOS_PER_DAY = 86_400 * 10**9


def _extract_level_rows(df: pd.DataFrame) -> list[_LevelRow]:
    """Pre-extract DataFrame rows into lightweight namedtuples."""
    rows: list[_LevelRow] = []
    for r in df.itertuples(index=False):
        sd = pd.Timestamp(r.swing_date)
        ao = pd.Timestamp(r.available_on)
        rows.append(_LevelRow(
            value=float(r.value),
            swing_date_ns=sd.value,
            available_on_ns=ao.value,
            level_score=float(r.level_score),
            swing_tier=getattr(r, "swing_tier", None),
            structure_label=getattr(r, "structure_label", None),
            swing_date_ts=sd,
            available_on_ts=ao,
        ))
    return rows


def _candidate_indices(
    avail_ns: np.ndarray,
    score_order: np.ndarray,
    current_ts_ns: int,
    cutoff_ns: int | None,
    top_n: int,
) -> tuple[int, ...]:
    """Return sorted positional indices of the candidate pool.

    Equivalent to ``_candidate_pool_for_fib`` but uses bisect on pre-sorted
    arrays instead of per-call pandas filter + sort.
    """
    upper = bisect_right(avail_ns, current_ts_ns)
    lower = bisect_left(avail_ns, cutoff_ns) if cutoff_ns is not None else 0
    if upper <= lower:
        return ()
    # Recency top_n: last top_n positions in the window
    recency_start = max(lower, upper - top_n)
    indices = set(range(recency_start, upper))
    # Score top_n: scan pre-computed argsort, pick first top_n in window
    count = 0
    for idx in score_order:
        if lower <= idx < upper:
            indices.add(idx)
            count += 1
            if count >= top_n:
                break
    return tuple(sorted(indices))


def _score_pair_fast(
    high: _LevelRow,
    low: _LevelRow,
    current_ts_ns: int,
    settings: dict[str, Any],
) -> tuple[float, str] | None:
    """Score a (high, low) pair — same logic as ``_score_fib_pair`` but
    operates on pre-extracted namedtuples with int64-ns timestamps."""
    if not np.isfinite(high.value) or not np.isfinite(low.value) or high.value <= low.value:
        return None
    if high.swing_date_ns == low.swing_date_ns:
        return None
    leg_direction = "bullish" if high.swing_date_ns > low.swing_date_ns else "bearish"

    newer_avail = max(high.available_on_ns, low.available_on_ns)
    older_avail = min(high.available_on_ns, low.available_on_ns)
    newer_age_days = max(0.0, (current_ts_ns - newer_avail) / _NANOS_PER_DAY)
    older_age_days = max(0.0, (current_ts_ns - older_avail) / _NANOS_PER_DAY)
    swing_span_days = abs(high.swing_date_ns - low.swing_date_ns) / _NANOS_PER_DAY

    score = high.level_score + low.level_score
    score += _tier_bonus(high.swing_tier or "") + _tier_bonus(low.swing_tier or "")
    if leg_direction == "bullish" and high.structure_label in ("HH", "EQH"):
        score += 0.75
    if leg_direction == "bearish" and low.structure_label in ("LL", "EQL"):
        score += 0.75
    penalty = float(settings["recency_penalty"])
    score -= newer_age_days * penalty
    score -= older_age_days * penalty * 0.20

    lookback_days = settings["lookback_days"]
    if lookback_days is not None and swing_span_days > float(lookback_days):
        score -= (swing_span_days - float(lookback_days)) * penalty * 0.10

    return score, leg_direction


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

    empty_cols_numeric = [
        f"{prefix}_anchor_low", f"{prefix}_anchor_high", f"{prefix}_range",
        f"{prefix}_pair_score", f"{prefix}_leg_position",
    ]
    empty_cols_object = [
        f"{prefix}_anchor_low_date", f"{prefix}_anchor_high_date",
        f"{prefix}_leg_direction", f"{prefix}_pair_key",
    ]
    if highs.empty or lows.empty:
        # Build all stub columns in a single dict, then concat once to
        # avoid DataFrame fragmentation from repeated single-column inserts.
        n = len(base)
        stubs: dict[str, Any] = {}
        for col in empty_cols_numeric:
            stubs[col] = np.full(n, np.nan)
        for col in empty_cols_object:
            stubs[col] = np.full(n, None, dtype=object)
        stubs[f"{prefix}_anchor_changed_flag"] = np.zeros(n, dtype=bool)
        stubs[f"{prefix}_direction_changed_flag"] = np.zeros(n, dtype=bool)
        for ratio in FIB_BASE_RATIOS:
            label = fib_ratio_label(ratio)
            for suffix in (f"_{label}_level", f"_distance_to_{label}_pct",
                           f"_mirror_{label}_level", f"_directional_{label}_level",
                           f"_directional_distance_to_{label}_pct"):
                stubs[f"{prefix}{suffix}"] = np.full(n, np.nan)
        for ratio in FIB_EXTENSION_RATIOS:
            label = fib_ratio_label(ratio)
            for suffix in (f"_ext_up_{label}_level", f"_ext_down_{label}_level",
                           f"_ext_{label}_level", f"_distance_to_ext_{label}_pct",
                           f"_distance_to_ext_up_{label}_pct", f"_distance_to_ext_down_{label}_pct"):
                stubs[f"{prefix}{suffix}"] = np.full(n, np.nan)
        for side in ("high", "low"):
            stubs[f"{prefix}_{side}_available_on"] = pd.array([pd.NaT] * n, dtype="datetime64[ns, UTC]")
            stubs[f"{prefix}_anchor_{side}_date"] = pd.array([pd.NaT] * n, dtype="datetime64[ns, UTC]")
            stubs[f"{prefix}_{side}_score"] = np.full(n, np.nan)
            stubs[f"{prefix}_{side}_label"] = np.full(n, None, dtype=object)
            stubs[f"{prefix}_{side}_tier"] = np.full(n, None, dtype=object)
        stub_df = pd.DataFrame(stubs, index=base.index)
        return pd.concat([base, stub_df], axis=1)

    work = base.sort_values("close_time").reset_index(drop=True).copy()

    # --- Optimized pair selection ----------------------------------------
    # Pre-extract level data into plain arrays / namedtuples so the hot
    # loop avoids per-timestamp DataFrame sorts entirely.
    settings = _fib_scope_settings(scope)
    top_n = int(settings["top_n"])
    lookback_days = settings["lookback_days"]

    high_rows = _extract_level_rows(highs)
    low_rows = _extract_level_rows(lows)

    highs_avail_ns = pd.DatetimeIndex(highs["available_on"]).asi8
    lows_avail_ns = pd.DatetimeIndex(lows["available_on"]).asi8

    high_scores = np.array([r.level_score for r in high_rows])
    high_score_order = np.lexsort((-highs_avail_ns, -high_scores))
    low_scores = np.array([r.level_score for r in low_rows])
    low_score_order = np.lexsort((-lows_avail_ns, -low_scores))

    lookback_ns = int(pd.Timedelta(days=int(lookback_days)).value) if lookback_days is not None else None
    timestamps_ns = pd.DatetimeIndex(work["close_time"]).asi8

    prev_hi_idx: tuple[int, ...] | None = None
    prev_lo_idx: tuple[int, ...] | None = None
    prev_best_pair: tuple[int, int] | None = None  # (hi, lo) indices of winner

    pair_rows: list[dict[str, Any]] = []
    last_pair: dict[str, Any] | None = None

    for ts_ns in timestamps_ns:
        cutoff_ns = (ts_ns - lookback_ns) if lookback_ns is not None else None

        hi_idx = _candidate_indices(highs_avail_ns, high_score_order, ts_ns, cutoff_ns, top_n)
        lo_idx = _candidate_indices(lows_avail_ns, low_score_order, ts_ns, cutoff_ns, top_n)

        if not hi_idx or not lo_idx:
            selected = None
        elif hi_idx == prev_hi_idx and lo_idx == prev_lo_idx and prev_best_pair is not None:
            # Pool unchanged — pair ranking is preserved (uniform age shift),
            # so just recompute the score for the same winning pair.
            bhi, blo = prev_best_pair
            scored = _score_pair_fast(high_rows[bhi], low_rows[blo], ts_ns, settings)
            if scored is not None:
                selected = last_pair.copy() if last_pair is not None else None
                if selected is not None:
                    selected["pair_score_adjusted"] = scored[0]
            else:
                selected = None
        else:
            # Pool changed — full pair scoring
            best: dict[str, Any] | None = None
            best_hi_lo: tuple[int, int] | None = None
            for hi in hi_idx:
                h = high_rows[hi]
                for lo in lo_idx:
                    l = low_rows[lo]
                    scored = _score_pair_fast(h, l, ts_ns, settings)
                    if scored is None:
                        continue
                    pair_score, leg_dir = scored
                    candidate = {
                        "pair_score_adjusted": pair_score,
                        "leg_direction": leg_dir,
                        "anchor_high_date": h.swing_date_ts,
                        "anchor_high": h.value,
                        "high_score": h.level_score,
                        "high_label": h.structure_label,
                        "high_tier": h.swing_tier,
                        "anchor_low_date": l.swing_date_ts,
                        "anchor_low": l.value,
                        "low_score": l.level_score,
                        "low_label": l.structure_label,
                        "low_tier": l.swing_tier,
                        "pair_key": (
                            f"{l.swing_date_ts.date()}|{l.value:.8f}"
                            f"__{h.swing_date_ts.date()}|{h.value:.8f}"
                        ),
                        "pair_available_on": max(h.available_on_ts, l.available_on_ts),
                    }
                    if best is None or pair_score > best["pair_score_adjusted"]:
                        best = candidate
                        best_hi_lo = (hi, lo)
                    elif pair_score == best["pair_score_adjusted"] and candidate["pair_available_on"] > best["pair_available_on"]:
                        best = candidate
                        best_hi_lo = (hi, lo)
            selected = best
            prev_best_pair = best_hi_lo

        prev_hi_idx = hi_idx
        prev_lo_idx = lo_idx
        if selected is not None:
            last_pair = selected
            if prev_best_pair is None and hi_idx and lo_idx:
                # Shouldn't happen, but guard against it
                pass
        pair_rows.append(last_pair.copy() if last_pair is not None else {})

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        pair_df = pd.DataFrame(index=work.index)

    defaults: list[tuple[str, Any]] = [
        ("pair_score_adjusted", np.nan), ("leg_direction", None),
        ("anchor_high_date", pd.NaT), ("anchor_high", np.nan),
        ("high_score", np.nan), ("high_label", None), ("high_tier", None),
        ("anchor_low_date", pd.NaT), ("anchor_low", np.nan),
        ("low_score", np.nan), ("low_label", None), ("low_tier", None),
        ("pair_key", None), ("pair_available_on", pd.NaT),
    ]
    for col, default in defaults:
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
        work[col] = np.where(active_pair, work[f"{prefix}_anchor_low"] + work[f"{prefix}_range"] * ratio, np.nan)
        work[f"{prefix}_distance_to_{label}_pct"] = np.where(work[col].notna(), work["close"] / work[col] - 1.0, np.nan)
        mirror_label = fib_ratio_label(_mirror_ratio(ratio))
        mirror_col = f"{prefix}_mirror_{label}_level"
        work[mirror_col] = np.where(active_pair, work[f"{prefix}_anchor_low"] + work[f"{prefix}_range"] * _mirror_ratio(ratio), np.nan)
        work[f"{prefix}_directional_{label}_level"] = np.where(
            active_pair & work[f"{prefix}_leg_direction"].eq("bullish"), work[col],
            np.where(active_pair & work[f"{prefix}_leg_direction"].eq("bearish"), work[mirror_col], np.nan),
        )
        work[f"{prefix}_directional_distance_to_{label}_pct"] = np.where(
            work[f"{prefix}_directional_{label}_level"].notna(),
            work["close"] / work[f"{prefix}_directional_{label}_level"] - 1.0,
            np.nan,
        )

    for ratio in FIB_EXTENSION_RATIOS:
        label = fib_ratio_label(ratio)
        up_col = f"{prefix}_ext_up_{label}_level"
        down_col = f"{prefix}_ext_down_{label}_level"
        ext_col = f"{prefix}_ext_{label}_level"
        work[up_col] = np.where(active_pair, work[f"{prefix}_anchor_high"] + work[f"{prefix}_range"] * (ratio - 1.0), np.nan)
        work[down_col] = np.where(active_pair, work[f"{prefix}_anchor_low"] - work[f"{prefix}_range"] * (ratio - 1.0), np.nan)
        work[ext_col] = np.where(
            active_pair & work[f"{prefix}_leg_direction"].eq("bullish"), work[up_col],
            np.where(active_pair & work[f"{prefix}_leg_direction"].eq("bearish"), work[down_col], np.nan),
        )
        work[f"{prefix}_distance_to_ext_{label}_pct"] = np.where(work[ext_col].notna(), work["close"] / work[ext_col] - 1.0, np.nan)
        work[f"{prefix}_distance_to_ext_up_{label}_pct"] = np.where(work[up_col].notna(), work["close"] / work[up_col] - 1.0, np.nan)
        work[f"{prefix}_distance_to_ext_down_{label}_pct"] = np.where(work[down_col].notna(), work["close"] / work[down_col] - 1.0, np.nan)

    return work


# ---------------------------------------------------------------------------
# Zone / cross-scope features
# ---------------------------------------------------------------------------

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
        close, base[f"{prefix}_directional_0_618_level"], base[f"{prefix}_directional_0_66_level"],
    )
    base[f"{prefix}_directional_0_34_0_382_zone_flag"] = _between_series(
        close, base[f"{prefix}_directional_0_34_level"], base[f"{prefix}_directional_0_382_level"],
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


# ---------------------------------------------------------------------------
# Column → scope dependency resolution
# ---------------------------------------------------------------------------

_BASE_FEATURE_COLUMNS = [
    "close_time", "open", "high", "low", "close", "volume", "atr",
    "market_bias_asof", "market_bias_after_close",
    "structure_break_event_on_close",
    "bos_up_on_close_flag", "bos_down_on_close_flag",
    "choch_up_on_close_flag", "choch_down_on_close_flag",
    "choch_any_on_close_flag", "bos_any_on_close_flag",
    "confirmed_high_on_close_flag", "confirmed_low_on_close_flag",
    "confirmed_high_label_on_close", "confirmed_low_label_on_close",
]
_ALL_SCOPES = ("local", "structural", "major", "global")
_BREAK_STATE_SUFFIXES = (
    "last_break_is_bullish",
    "last_break_is_bearish",
    "last_break_is_choch_up",
    "last_break_is_choch_down",
    "last_break_is_bos_up",
    "last_break_is_bos_down",
)
_BREAK_EVENT_NAMES = ("bos_up", "bos_down", "choch_up", "choch_down")
_CROSS_SCOPE_COLUMNS = {
    "major_global_break_alignment_bullish",
    "major_global_break_alignment_bearish",
    "major_global_fib_alignment_bullish",
    "major_global_fib_alignment_bearish",
    "major_global_bullish_confluence_flag",
    "major_global_bearish_confluence_flag",
    "major_global_break_disagreement_flag",
    "major_global_fib_disagreement_flag",
    "major_pullback_long_candidate_flag",
    "major_pullback_short_candidate_flag",
    "global_continuation_long_flag",
    "global_continuation_short_flag",
}


def _requested_columns_set(columns: list[str] | None) -> set[str] | None:
    if columns is None:
        return None
    return set(columns)


def _scopes_with_requested_prefixes(
    requested: set[str] | None,
    prefixes_by_scope: dict[str, tuple[str, ...]],
) -> set[str]:
    if requested is None:
        return set(prefixes_by_scope)
    scopes: set[str] = set()
    for scope, prefixes in prefixes_by_scope.items():
        if any(any(col.startswith(prefix) for prefix in prefixes) for col in requested):
            scopes.add(scope)
    return scopes


def _requested_break_merge_scopes(requested: set[str] | None) -> set[str]:
    if requested is None:
        return set(_ALL_SCOPES)
    scopes: set[str] = set()
    cross_scope_requested = bool(requested & _CROSS_SCOPE_COLUMNS)
    for scope in _ALL_SCOPES:
        if any(col.startswith(f"{scope}_break_") for col in requested):
            scopes.add(scope)
        if any(col == f"{scope}_{suffix}" for col in requested for suffix in _BREAK_STATE_SUFFIXES):
            scopes.add(scope)
    if cross_scope_requested:
        scopes.update({"major", "global"})
    return scopes


def _requested_break_state_scopes(requested: set[str] | None) -> set[str]:
    if requested is None:
        return set(_ALL_SCOPES)
    scopes: set[str] = set()
    cross_scope_requested = bool(requested & _CROSS_SCOPE_COLUMNS)
    for scope in _ALL_SCOPES:
        if any(col == f"{scope}_{suffix}" for col in requested for suffix in _BREAK_STATE_SUFFIXES):
            scopes.add(scope)
    if cross_scope_requested:
        scopes.update({"major", "global"})
    return scopes


def _requested_break_event_scopes(requested: set[str] | None) -> dict[str, tuple[str, ...]]:
    if requested is None:
        return {scope: _BREAK_EVENT_NAMES for scope in _ALL_SCOPES}
    scoped_events: dict[str, list[str]] = {}
    for scope in _ALL_SCOPES:
        events = [
            event_name
            for event_name in _BREAK_EVENT_NAMES
            if any(
                col in {
                    f"{scope}_{event_name}_available_on",
                    f"{scope}_{event_name}_score",
                }
                for col in requested
            )
        ]
        if events:
            scoped_events[scope] = events
    return {scope: tuple(events) for scope, events in scoped_events.items()}


def _requested_level_merge_scopes(requested: set[str] | None) -> set[str]:
    prefixes = {
        scope: (
            f"{scope}_high_",
            f"{scope}_low_",
            f"{scope}_distance_to_high_pct",
            f"{scope}_distance_to_low_pct",
            f"{scope}_box_position",
        )
        for scope in _ALL_SCOPES
    }
    return _scopes_with_requested_prefixes(requested, prefixes)


def _requested_rolling_event_scopes(requested: set[str] | None) -> set[str]:
    prefixes = {
        scope: (
            f"{scope}_up_events_",
            f"{scope}_down_events_",
            f"{scope}_net_break_pressure_",
        )
        for scope in _ALL_SCOPES
    }
    return _scopes_with_requested_prefixes(requested, prefixes)


def _select_feature_output_columns(
    feature_matrix: pd.DataFrame,
    columns: list[str] | None,
) -> pd.DataFrame:
    if columns is None:
        return feature_matrix
    keep: list[str] = []
    for col in _BASE_FEATURE_COLUMNS:
        if col in feature_matrix.columns and col not in keep:
            keep.append(col)
    missing = [col for col in columns if col not in feature_matrix.columns]
    if missing:
        raise KeyError(f"Requested BTC structure columns were not built: {missing}")
    for col in columns:
        if col not in keep:
            keep.append(col)
    return feature_matrix.loc[:, keep]


def derive_fib_scopes(columns: list[str] | None) -> tuple[str, ...]:
    """Determine which fib scopes must be computed for *columns*.

    Returns an empty tuple when no fib columns are requested (e.g.
    ``STRUCTURE_EVENTS`` only), which skips the expensive O(N×P²) pair
    selection entirely.
    """
    if columns is None:
        return ("local", "major", "global")
    scopes: set[str] = set()
    for col in columns:
        if "local_fib_" in col:
            scopes.add("local")
        if "major_fib_" in col or "major_pullback_" in col:
            scopes.add("major")
        if "global_fib_" in col or "global_continuation_" in col:
            scopes.add("global")
        if col.startswith("major_global_"):
            scopes.update(("major", "global"))
    return tuple(sorted(scopes))


def _needs_ranking(columns: list[str] | None) -> bool:
    """True if any column requires ranked levels / breaks."""
    if columns is None:
        return True
    events_only = set(STRUCTURE_EVENTS)
    return not set(columns).issubset(events_only | {"market_bias_after_close"})


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_structure_feature_matrix(
    structure: StructureArtifacts,
    *,
    fib_scopes: tuple[str, ...] = ("local", "major", "global"),
    skip_ranking: bool = False,
    ranked_highs: pd.DataFrame | None = None,
    ranked_lows: pd.DataFrame | None = None,
    ranked_breaks: pd.DataFrame | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    requested = _requested_columns_set(columns)
    base = structure.features[_BASE_FEATURE_COLUMNS].copy()
    base["close_time"] = pd.to_datetime(base["close_time"], utc=True)

    if skip_ranking:
        base = base.sort_values("close_time").reset_index(drop=True)
        return _select_feature_output_columns(base, columns)

    if ranked_highs is None:
        ranked_highs = rank_confirmed_levels(structure.confirmed_highs, kind="high")
    if ranked_lows is None:
        ranked_lows = rank_confirmed_levels(structure.confirmed_lows, kind="low")
    if ranked_breaks is None:
        ranked_breaks = rank_structure_breaks(structure.structure_breaks, structure.confirmed_highs, structure.confirmed_lows)

    level_merge_scopes = _requested_level_merge_scopes(requested)
    break_merge_scopes = _requested_break_merge_scopes(requested)
    rolling_event_scopes = _requested_rolling_event_scopes(requested)
    break_state_scopes = _requested_break_state_scopes(requested)
    break_event_scopes = _requested_break_event_scopes(requested)

    for scope in sorted(level_merge_scopes):
        base = _merge_last_levels(base, ranked_highs, kind="high", scope=scope)
        base = _merge_last_levels(base, ranked_lows, kind="low", scope=scope)
        base = _attach_distance_features(base, scope=scope)

    for scope in sorted(break_merge_scopes):
        base = _merge_last_breaks(base, ranked_breaks, scope=scope)

    if rolling_event_scopes:
        base = _rolling_event_counts(base, ranked_breaks, scopes=tuple(sorted(rolling_event_scopes)))

    for scope in fib_scopes:
        base = _build_fib_leg_features(base, ranked_highs, ranked_lows, scope=scope)

    if break_state_scopes or break_event_scopes:
        base = base.copy()
    for scope in sorted(break_state_scopes):
        base = _attach_scope_break_state_features(base, scope=scope)
    for scope, event_names in break_event_scopes.items():
        for event_name in event_names:
            base = _merge_last_break_event(base, ranked_breaks, scope=scope, event_name=event_name)

    if fib_scopes:
        base = base.copy()
    for scope in fib_scopes:
        base = _attach_scope_fib_zone_features(base, scope=scope)

    base = base.copy()
    if "major" in fib_scopes and "global" in fib_scopes:
        if requested is None or bool(requested & _CROSS_SCOPE_COLUMNS):
            base = _attach_cross_scope_structure_features(base)
    base = base.sort_values("close_time").reset_index(drop=True)
    return _select_feature_output_columns(base, columns)


# ---------------------------------------------------------------------------
# Lab runner
# ---------------------------------------------------------------------------

def run_structure_feature_lab(
    structure: StructureArtifacts,
    *,
    columns: list[str] | None = None,
) -> StructureLabArtifacts:
    """Run the ranking + feature matrix pipeline.

    Parameters
    ----------
    columns:
        If given, only compute the feature layers required to produce
        these columns.  Fib scopes that aren't needed are skipped,
        which avoids the expensive O(N×P²) pair selection.

        Pass ``None`` to compute everything (full 510+ column matrix).

        Typical usage::

            lab = run_structure_feature_lab(structure, columns=STRUCTURE_REGIME)
    """
    fib_scopes = derive_fib_scopes(columns)
    skip_ranking = not _needs_ranking(columns)

    if skip_ranking:
        ranked_highs = pd.DataFrame()
        ranked_lows = pd.DataFrame()
        ranked_breaks = pd.DataFrame()
    else:
        ranked_highs = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        ranked_lows = rank_confirmed_levels(structure.confirmed_lows, kind="low")
        ranked_breaks = rank_structure_breaks(structure.structure_breaks, structure.confirmed_highs, structure.confirmed_lows)

    feature_matrix = build_structure_feature_matrix(
        structure,
        fib_scopes=fib_scopes,
        skip_ranking=skip_ranking,
        ranked_highs=ranked_highs,
        ranked_lows=ranked_lows,
        ranked_breaks=ranked_breaks,
        columns=columns,
    )
    return StructureLabArtifacts(
        ranked_highs=ranked_highs,
        ranked_lows=ranked_lows,
        ranked_breaks=ranked_breaks,
        feature_matrix=feature_matrix,
        summary={
            "ranked_highs": len(ranked_highs),
            "ranked_lows": len(ranked_lows),
            "ranked_breaks": len(ranked_breaks),
            "feature_rows": len(feature_matrix),
            "feature_columns": list(feature_matrix.columns),
            "fib_scopes_computed": list(fib_scopes),
        },
    )


# ---------------------------------------------------------------------------
# Pre-defined column sets for strategy consumption
# ---------------------------------------------------------------------------

STRUCTURE_REGIME = [
    "market_bias_after_close",
    "major_global_bullish_confluence_flag",
    "major_global_bearish_confluence_flag",
    "global_continuation_long_flag",
    "global_continuation_short_flag",
    "major_last_break_is_bullish",
    "global_last_break_is_bullish",
]

STRUCTURE_LEVELS = [
    "major_fib_leg_direction",
    "major_fib_leg_position",
    "major_fib_0_5_level",
    "major_fib_0_618_level",
    "major_fib_0_66_level",
    "major_fib_0_34_level",
    "major_fib_ext_up_1_618_level",
    "major_pullback_long_candidate_flag",
    "major_pullback_short_candidate_flag",
    "global_fib_leg_position",
]

STRUCTURE_EVENTS = [
    "choch_up_on_close_flag",
    "choch_down_on_close_flag",
    "bos_up_on_close_flag",
    "bos_down_on_close_flag",
    "confirmed_high_on_close_flag",
    "confirmed_low_on_close_flag",
]
