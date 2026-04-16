from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .btc_structure import (
    StructureArtifacts,
    build_structure_figure,
    rank_confirmed_levels,
    rank_structure_breaks,
)


@dataclass(slots=True)
class StructureLabArtifacts:
    ranked_highs: pd.DataFrame
    ranked_lows: pd.DataFrame
    ranked_breaks: pd.DataFrame
    feature_matrix: pd.DataFrame
    summary: dict[str, Any]


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


def build_scope_filtered_figure(
    artifacts: StructureArtifacts,
    *,
    level_scopes: tuple[str, ...] | None = None,
    break_scopes: tuple[str, ...] | None = None,
    break_events: tuple[str, ...] | None = None,
    min_level_score: float | None = None,
    only_strategy_levels: bool = False,
    only_strategy_breaks: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    title: str | None = None,
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
        confirmed_highs=filter_ranked_levels(
            ranked_highs,
            scopes=level_scopes,
            min_level_score=min_level_score,
            only_strategy_levels=only_strategy_levels,
        ),
        confirmed_lows=filter_ranked_levels(
            ranked_lows,
            scopes=level_scopes,
            min_level_score=min_level_score,
            only_strategy_levels=only_strategy_levels,
        ),
        break_attempt_highs=artifacts.break_attempt_highs.iloc[0:0].copy(),
        break_attempt_lows=artifacts.break_attempt_lows.iloc[0:0].copy(),
        structure_breaks=filter_ranked_breaks(
            ranked_breaks,
            scopes=break_scopes,
            events=break_events,
            min_level_score=min_level_score,
            only_strategy_breaks=only_strategy_breaks,
        ),
        summary=artifacts.summary,
    )
    return build_structure_figure(
        filtered,
        title=title,
        start_date=start_date,
        end_date=end_date,
        include_candidates=False,
        include_break_attempts=False,
        include_structure_breaks=True,
    )


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
        ["available_on", "swing_date", "value", "structure_label", "level_score", "level_priority", "max_window"]
    ].rename(
        columns={
            "available_on": f"{prefix}_available_on",
            "swing_date": f"{prefix}_swing_date",
            "value": f"{prefix}_value",
            "structure_label": f"{prefix}_label",
            "level_score": f"{prefix}_score",
            "level_priority": f"{prefix}_priority",
            "max_window": f"{prefix}_max_window",
        }
    )
    merged = pd.merge_asof(
        base.sort_values("date"),
        part,
        left_on="date",
        right_on=f"{prefix}_available_on",
        direction="backward",
    )
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
    part = part.sort_values("available_on")[
        ["available_on", "event", "broken_level_score", "broken_level_priority", "broken_level_max_window"]
    ].rename(
        columns={
            "available_on": f"{prefix}_available_on",
            "event": f"{prefix}_event",
            "broken_level_score": f"{prefix}_score",
            "broken_level_priority": f"{prefix}_priority",
            "broken_level_max_window": f"{prefix}_max_window",
        }
    )
    merged = pd.merge_asof(
        base.sort_values("date"),
        part,
        left_on="date",
        right_on=f"{prefix}_available_on",
        direction="backward",
    )
    delta = merged["date"] - merged[f"{prefix}_available_on"]
    merged[f"{prefix}_days_since"] = delta.dt.total_seconds() / 86400.0
    return merged


def _attach_distance_features(base: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    high_col = f"{scope}_high_value"
    low_col = f"{scope}_low_value"
    base[f"{scope}_distance_to_high_pct"] = np.where(
        base[high_col].notna(),
        base["close"] / base[high_col] - 1.0,
        np.nan,
    )
    base[f"{scope}_distance_to_low_pct"] = np.where(
        base[low_col].notna(),
        base["close"] / base[low_col] - 1.0,
        np.nan,
    )
    width = base[high_col] - base[low_col]
    base[f"{scope}_box_position"] = np.where(
        width.abs() > 1e-9,
        (base["close"] - base[low_col]) / width,
        np.nan,
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
    merged = merged.sort_values("date")
    event_cols = [column for column in merged.columns if column != "date"]
    for column in event_cols:
        merged[column] = merged[column].astype(float)
    for scope in scopes:
        up_cols = [f"{scope}__bos_up", f"{scope}__choch_up"]
        down_cols = [f"{scope}__bos_down", f"{scope}__choch_down"]
        for window in windows:
            existing_up = [col for col in up_cols if col in merged.columns]
            existing_down = [col for col in down_cols if col in merged.columns]
            merged[f"{scope}_up_events_{window}"] = (
                merged[existing_up].sum(axis=1).rolling(window=window, min_periods=1).sum()
                if existing_up
                else 0.0
            )
            merged[f"{scope}_down_events_{window}"] = (
                merged[existing_down].sum(axis=1).rolling(window=window, min_periods=1).sum()
                if existing_down
                else 0.0
            )
            merged[f"{scope}_net_break_pressure_{window}"] = (
                merged[f"{scope}_up_events_{window}"] - merged[f"{scope}_down_events_{window}"]
            )
    keep_cols = ["date"] + [column for column in merged.columns if column != "date" and "__" not in column]
    return base.merge(merged[keep_cols], on="date", how="left")


def build_structure_feature_matrix(artifacts: StructureArtifacts) -> pd.DataFrame:
    ranked_highs = rank_confirmed_levels(artifacts.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(artifacts.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(
        artifacts.structure_breaks,
        artifacts.confirmed_highs,
        artifacts.confirmed_lows,
    )

    base = artifacts.features[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "market_bias_asof",
            "market_bias_after_close",
            "structure_break_event_on_close",
            "choch_any_on_close_flag",
            "bos_any_on_close_flag",
        ]
    ].copy()
    base["date"] = pd.to_datetime(base["date"], utc=True)

    scopes = ("local", "structural", "major", "global")
    for scope in scopes:
        base = _merge_last_levels(base, ranked_highs, kind="high", scope=scope)
        base = _merge_last_levels(base, ranked_lows, kind="low", scope=scope)
        base = _merge_last_breaks(base, ranked_breaks, scope=scope)
        base = _attach_distance_features(base, scope=scope)

    base = _rolling_event_counts(base, ranked_breaks, scopes=("local", "structural", "major", "global"))
    return base.sort_values("date").reset_index(drop=True)


def summarize_structure_lab(lab: StructureLabArtifacts) -> dict[str, Any]:
    ranked_breaks = lab.ranked_breaks
    feature_matrix = lab.feature_matrix
    scope_counts = ranked_breaks["broken_level_scope"].value_counts().to_dict() if not ranked_breaks.empty else {}
    strategy_counts = (
        ranked_breaks[ranked_breaks["is_strategy_break"].fillna(False)]["event"].value_counts().to_dict()
        if not ranked_breaks.empty
        else {}
    )
    return {
        "ranked_highs": int(len(lab.ranked_highs)),
        "ranked_lows": int(len(lab.ranked_lows)),
        "ranked_breaks": int(len(lab.ranked_breaks)),
        "break_scope_counts": {str(key): int(value) for key, value in scope_counts.items()},
        "strategy_break_counts": {str(key): int(value) for key, value in strategy_counts.items()},
        "feature_rows": int(len(feature_matrix)),
        "feature_columns": list(feature_matrix.columns),
    }


def run_structure_feature_lab(artifacts: StructureArtifacts) -> StructureLabArtifacts:
    ranked_highs = rank_confirmed_levels(artifacts.confirmed_highs, kind="high")
    ranked_lows = rank_confirmed_levels(artifacts.confirmed_lows, kind="low")
    ranked_breaks = rank_structure_breaks(
        artifacts.structure_breaks,
        artifacts.confirmed_highs,
        artifacts.confirmed_lows,
    )
    feature_matrix = build_structure_feature_matrix(artifacts)
    lab = StructureLabArtifacts(
        ranked_highs=ranked_highs,
        ranked_lows=ranked_lows,
        ranked_breaks=ranked_breaks,
        feature_matrix=feature_matrix,
        summary={},
    )
    lab.summary = summarize_structure_lab(lab)
    return lab


def save_structure_feature_lab(lab: StructureLabArtifacts, output_dir: Path | str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lab.ranked_highs.to_csv(output_path / "confirmed_highs_ranked.csv", index=False)
    lab.ranked_lows.to_csv(output_path / "confirmed_lows_ranked.csv", index=False)
    lab.ranked_breaks.to_csv(output_path / "structure_breaks_ranked.csv", index=False)
    lab.feature_matrix.to_csv(output_path / "structure_feature_matrix.csv", index=False)
    with (output_path / "structure_feature_lab_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(lab.summary, handle, indent=2, ensure_ascii=False)
    return output_path
