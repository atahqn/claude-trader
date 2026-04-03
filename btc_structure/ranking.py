from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def _classify_level_scope(max_window: int) -> str:
    if max_window >= 180:
        return "global"
    if max_window >= 90:
        return "major"
    if max_window >= 30:
        return "structural"
    return "local"


def _window_score(max_window: int) -> int:
    if max_window >= 365:
        return 5
    if max_window >= 180:
        return 4
    if max_window >= 90:
        return 3
    if max_window >= 30:
        return 2
    if max_window >= 7:
        return 1
    return 0


def _classify_swing_tier(max_window: int, confluence_count: int, confluence_long_hits: int) -> str:
    if confluence_count >= 4 and confluence_long_hits >= 2 and max_window >= 180:
        return "global_extrema"
    if confluence_count >= 3 and confluence_long_hits >= 1 and max_window >= 30:
        return "structural_extrema"
    return "local_extrema"


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
    ranked["label_bonus"] = ranked["structure_label"].map({
        "HH": 2 if kind == "high" else 0,
        "LL": 2 if kind == "low" else 0,
        "HL": 1 if kind == "low" else 0,
        "LH": 1 if kind == "high" else 0,
        "EQH": 1 if kind == "high" else 0,
        "EQL": 1 if kind == "low" else 0,
    }).fillna(0)
    ranked["level_score"] = (
        ranked["confluence_count"].fillna(0).astype(int)
        + ranked["window_score"]
        + ranked["structure_bonus"]
        + ranked["label_bonus"]
    )
    ranked["swing_tier"] = [
        _classify_swing_tier(mw, cc, cl)
        for mw, cc, cl in zip(
            ranked["max_window"].fillna(0),
            ranked["confluence_count"].fillna(0),
            ranked["confluence_long_hits"].fillna(0),
            strict=False,
        )
    ]
    ranked["level_priority"] = (
        pd.cut(ranked["level_score"], bins=[-1, 5, 8, 11, 100], labels=["low", "medium", "high", "critical"])
        .astype(str)
    )
    ranked["is_strategy_level"] = (
        ranked["level_scope"].isin(["structural", "major", "global"])
        & (ranked["level_score"] >= 7)
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
    scopes: list[str] = []
    scores: list[float] = []
    priorities: list[str] = []
    for _, row in ranked.iterrows():
        source = ranked_highs if row.get("level_kind") == "high" else ranked_lows
        value = row.get("level_value")
        when = pd.Timestamp(row["available_on"])
        if source.empty or pd.isna(value):
            scopes.append("unknown")
            scores.append(np.nan)
            priorities.append("unranked")
            continue
        matches = source[
            np.isclose(source["value"].astype(float), float(value), rtol=0.0, atol=1e-9)
            & (source["available_on"] <= when)
        ].sort_values("available_on")
        if matches.empty:
            scopes.append("unknown")
            scores.append(np.nan)
            priorities.append("unranked")
            continue
        match = matches.iloc[-1]
        scopes.append(str(match["level_scope"]))
        scores.append(float(match["level_score"]))
        priorities.append(str(match["level_priority"]))
    ranked["broken_level_scope"] = scopes
    ranked["broken_level_score"] = scores
    ranked["broken_level_priority"] = priorities
    ranked["is_major_break"] = ranked["broken_level_scope"].isin(["major", "global"])
    ranked["is_strategy_break"] = (
        ranked["event"].astype(str).str.startswith("choch")
        | (ranked["event"].astype(str).str.startswith("bos") & ranked["is_major_break"])
    )
    return ranked


def filter_ranked_levels(
    ranked_df: pd.DataFrame,
    *,
    scopes: tuple[str, ...] | None = None,
    min_level_score: float | None = None,
    priorities: tuple[str, ...] | None = None,
    labels: tuple[str, ...] | None = None,
    only_strategy_levels: bool = False,
) -> pd.DataFrame:
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


def filter_ranked_breaks(
    ranked_df: pd.DataFrame,
    *,
    scopes: tuple[str, ...] | None = None,
    events: tuple[str, ...] | None = None,
    min_level_score: float | None = None,
    priorities: tuple[str, ...] | None = None,
    only_strategy_breaks: bool = False,
) -> pd.DataFrame:
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
