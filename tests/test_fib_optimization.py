"""Verify the optimized _build_fib_leg_features produces identical results
to the original per-timestamp _select_active_fib_pair reference path.

The original functions (_candidate_pool_for_fib, _score_fib_pair,
_select_active_fib_pair) remain in features.py unchanged and serve as
the reference implementation.  The optimized loop inside
_build_fib_leg_features must produce bit-identical pair selections.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from btc_structure.config import BtcStructureConfig
from btc_structure.engine import simulate_btc_structure
from btc_structure.features import (
    _build_fib_leg_features,
    _candidate_indices,
    _candidate_pool_for_fib,
    _extract_level_rows,
    _fib_scope_settings,
    _filter_fib_source_levels,
    _score_fib_pair,
    _score_pair_fast,
    _select_active_fib_pair,
    run_structure_feature_lab,
)
from btc_structure.ranking import rank_confirmed_levels


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _make_trending_ohlcv(
    n: int = 300,
    start_price: float = 50000.0,
    trend: str = "up",
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [datetime(2022, 1, 1, 23, 59, 59, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    close = start_price
    drift = 0.003 if trend == "up" else -0.003
    rows = []
    for dt in dates:
        change_pct = rng.normal(drift, 0.015)
        open_ = close
        close = open_ * (1 + change_pct)
        high = max(open_, close) * (1 + abs(rng.normal(0.0, 0.004)))
        low = min(open_, close) * (1 - abs(rng.normal(0.0, 0.004)))
        volume = rng.uniform(1000, 5000)
        rows.append({
            "close_time": dt, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        })
    df = pd.DataFrame(rows)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


def _make_volatile_ohlcv(n: int = 400, seed: int = 77) -> pd.DataFrame:
    """Large swings to produce many confirmed highs/lows."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2021, 6, 1, 23, 59, 59, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    close = 40000.0
    rows = []
    for i, dt in enumerate(dates):
        # Alternate regime every 40 days
        phase = (i // 40) % 4
        drift = [0.005, -0.004, 0.002, -0.006][phase]
        change_pct = rng.normal(drift, 0.025)
        open_ = close
        close = open_ * (1 + change_pct)
        high = max(open_, close) * (1 + abs(rng.normal(0.0, 0.008)))
        low = min(open_, close) * (1 - abs(rng.normal(0.0, 0.008)))
        volume = rng.uniform(2000, 8000)
        rows.append({
            "close_time": dt, "open": open_, "high": high,
            "low": low, "close": close, "volume": volume,
        })
    df = pd.DataFrame(rows)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Helper: run the original (reference) pair selection per-timestamp
# ---------------------------------------------------------------------------

def _reference_pair_selection(
    base: pd.DataFrame,
    ranked_highs: pd.DataFrame,
    ranked_lows: pd.DataFrame,
    scope: str,
) -> list[dict]:
    """Use the original _select_active_fib_pair on each timestamp."""
    highs = _filter_fib_source_levels(ranked_highs, scope=scope)
    lows = _filter_fib_source_levels(ranked_lows, scope=scope)
    if highs.empty or lows.empty:
        return [{} for _ in range(len(base))]

    work = base.sort_values("close_time").reset_index(drop=True)
    pairs: list[dict] = []
    last_pair: dict | None = None
    for ts in pd.to_datetime(work["close_time"], utc=True):
        selected = _select_active_fib_pair(highs, lows, ts, scope=scope)
        if selected is not None:
            last_pair = selected
        pairs.append(last_pair.copy() if last_pair is not None else {})
    return pairs


# ---------------------------------------------------------------------------
# Helper: compare two pair dicts
# ---------------------------------------------------------------------------

def _assert_pairs_match(ref: dict, opt_row: pd.Series, prefix: str, row_idx: int):
    """Assert a reference pair dict matches the optimized DataFrame row."""
    if not ref:
        # Reference had no pair — optimized should have NaN/None
        pk = opt_row.get(f"{prefix}_pair_key")
        assert pk is None or (isinstance(pk, float) and np.isnan(pk)), (
            f"Row {row_idx}: expected no pair, got pair_key={pk}"
        )
        return

    # pair_key must match exactly
    opt_pk = opt_row[f"{prefix}_pair_key"]
    assert opt_pk == ref["pair_key"], (
        f"Row {row_idx}: pair_key mismatch: ref={ref['pair_key']!r} opt={opt_pk!r}"
    )

    # pair_score must be very close (same arithmetic, but int64-ns vs Timestamp path)
    opt_score = opt_row[f"{prefix}_pair_score"]
    ref_score = ref["pair_score_adjusted"]
    assert abs(opt_score - ref_score) < 1e-8, (
        f"Row {row_idx}: score mismatch: ref={ref_score:.12f} opt={opt_score:.12f}"
    )

    # anchor values
    assert abs(opt_row[f"{prefix}_anchor_high"] - ref["anchor_high"]) < 1e-8, (
        f"Row {row_idx}: anchor_high mismatch"
    )
    assert abs(opt_row[f"{prefix}_anchor_low"] - ref["anchor_low"]) < 1e-8, (
        f"Row {row_idx}: anchor_low mismatch"
    )

    # leg direction
    opt_dir = opt_row[f"{prefix}_leg_direction"]
    assert opt_dir == ref["leg_direction"], (
        f"Row {row_idx}: direction mismatch: ref={ref['leg_direction']} opt={opt_dir}"
    )


# ---------------------------------------------------------------------------
# Unit tests for _candidate_indices
# ---------------------------------------------------------------------------

class TestCandidateIndices:
    """Verify _candidate_indices matches _candidate_pool_for_fib."""

    def _compare_pools(self, highs, scope, timestamps):
        settings = _fib_scope_settings(scope)
        top_n = int(settings["top_n"])
        lookback_days = settings["lookback_days"]
        lookback_ns = int(pd.Timedelta(days=int(lookback_days)).value) if lookback_days is not None else None

        rows = _extract_level_rows(highs)
        avail_ns = pd.DatetimeIndex(highs["available_on"]).asi8
        scores = np.array([r.level_score for r in rows])
        score_order = np.lexsort((-avail_ns, -scores))

        for ts in timestamps:
            ts_ns = ts.value
            cutoff_ns = (ts_ns - lookback_ns) if lookback_ns is not None else None

            # Reference
            ref_pool = _candidate_pool_for_fib(highs, ts, settings=settings)
            ref_values = set(ref_pool["value"].round(8).tolist()) if not ref_pool.empty else set()

            # Optimized
            opt_idx = _candidate_indices(avail_ns, score_order, ts_ns, cutoff_ns, top_n)
            opt_values = {round(rows[i].value, 8) for i in opt_idx}

            assert ref_values == opt_values, (
                f"Pool mismatch at {ts}: ref={sorted(ref_values)} opt={sorted(opt_values)}"
            )

    def test_trending_up_local(self):
        ohlcv = _make_trending_ohlcv(n=200, trend="up")
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        ranked = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        highs = _filter_fib_source_levels(ranked, scope="local")
        if highs.empty:
            pytest.skip("No local highs in synthetic data")
        timestamps = pd.to_datetime(ohlcv["close_time"], utc=True)
        self._compare_pools(highs, "local", timestamps)

    def test_trending_up_major(self):
        ohlcv = _make_trending_ohlcv(n=200, trend="up")
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        ranked = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        highs = _filter_fib_source_levels(ranked, scope="major")
        if highs.empty:
            pytest.skip("No major highs in synthetic data")
        timestamps = pd.to_datetime(ohlcv["close_time"], utc=True)
        self._compare_pools(highs, "major", timestamps)

    def test_volatile_global(self):
        ohlcv = _make_volatile_ohlcv(n=300)
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        ranked = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        highs = _filter_fib_source_levels(ranked, scope="global")
        if highs.empty:
            pytest.skip("No global highs in synthetic data")
        timestamps = pd.to_datetime(ohlcv["close_time"], utc=True)
        self._compare_pools(highs, "global", timestamps)

    def test_lows_pool_matches(self):
        ohlcv = _make_volatile_ohlcv(n=300)
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        ranked = rank_confirmed_levels(structure.confirmed_lows, kind="low")
        lows = _filter_fib_source_levels(ranked, scope="major")
        if lows.empty:
            pytest.skip("No major lows in synthetic data")
        timestamps = pd.to_datetime(ohlcv["close_time"], utc=True)
        self._compare_pools(lows, "major", timestamps)


# ---------------------------------------------------------------------------
# Unit tests for _score_pair_fast
# ---------------------------------------------------------------------------

class TestScorePairFast:
    """Verify _score_pair_fast matches _score_fib_pair exactly."""

    def _compare_scores(self, highs, lows, scope, timestamps):
        settings = _fib_scope_settings(scope)
        high_rows = _extract_level_rows(highs)
        low_rows = _extract_level_rows(lows)

        for ts in timestamps[::10]:  # sample every 10th for speed
            pool_h = _candidate_pool_for_fib(highs, ts, settings=settings)
            pool_l = _candidate_pool_for_fib(lows, ts, settings=settings)
            if pool_h.empty or pool_l.empty:
                continue

            ts_ns = ts.value
            for hr in pool_h.itertuples(index=False):
                # Find the matching pre-extracted row
                hi_val = float(hr.value)
                hi_row = next(r for r in high_rows if abs(r.value - hi_val) < 1e-12)
                for lr in pool_l.itertuples(index=False):
                    lo_val = float(lr.value)
                    lo_row = next(r for r in low_rows if abs(r.value - lo_val) < 1e-12)

                    ref = _score_fib_pair(hr, lr, ts, settings=settings)
                    opt = _score_pair_fast(hi_row, lo_row, ts_ns, settings)

                    if ref is None:
                        assert opt is None, f"Expected None at {ts}, got {opt}"
                    else:
                        assert opt is not None, f"Expected score at {ts}, got None"
                        assert ref[1] == opt[1], f"Direction mismatch at {ts}"
                        assert abs(ref[0] - opt[0]) < 1e-8, (
                            f"Score mismatch at {ts}: ref={ref[0]:.12f} opt={opt[0]:.12f}"
                        )

    def test_trending_local(self):
        ohlcv = _make_trending_ohlcv(n=200, trend="up")
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        rh = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        rl = rank_confirmed_levels(structure.confirmed_lows, kind="low")
        highs = _filter_fib_source_levels(rh, scope="local")
        lows = _filter_fib_source_levels(rl, scope="local")
        if highs.empty or lows.empty:
            pytest.skip("No local levels")
        timestamps = pd.to_datetime(ohlcv["close_time"], utc=True)
        self._compare_scores(highs, lows, "local", timestamps)

    def test_volatile_major(self):
        ohlcv = _make_volatile_ohlcv(n=300)
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        rh = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        rl = rank_confirmed_levels(structure.confirmed_lows, kind="low")
        highs = _filter_fib_source_levels(rh, scope="major")
        lows = _filter_fib_source_levels(rl, scope="major")
        if highs.empty or lows.empty:
            pytest.skip("No major levels")
        timestamps = pd.to_datetime(ohlcv["close_time"], utc=True)
        self._compare_scores(highs, lows, "major", timestamps)


# ---------------------------------------------------------------------------
# Integration: full _build_fib_leg_features vs reference
# ---------------------------------------------------------------------------

class TestBuildFibLegFeaturesMatchesReference:
    """The optimized _build_fib_leg_features must produce identical pair
    selections as the original _select_active_fib_pair loop."""

    def _run_comparison(self, ohlcv, scopes=("local", "major", "global")):
        config = BtcStructureConfig()
        structure, _ = simulate_btc_structure(ohlcv, config)
        ranked_highs = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        ranked_lows = rank_confirmed_levels(structure.confirmed_lows, kind="low")

        base_cols = ["close_time", "open", "high", "low", "close", "volume", "atr"]
        base = structure.features[base_cols].copy()
        base["close_time"] = pd.to_datetime(base["close_time"], utc=True)

        for scope in scopes:
            highs = _filter_fib_source_levels(ranked_highs, scope=scope)
            lows = _filter_fib_source_levels(ranked_lows, scope=scope)
            if highs.empty or lows.empty:
                continue

            # Reference: original per-timestamp loop
            ref_pairs = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)

            # Optimized: the rewritten _build_fib_leg_features
            result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
            result = result.sort_values("close_time").reset_index(drop=True)

            prefix = f"{scope}_fib"
            assert len(ref_pairs) == len(result), (
                f"{scope}: row count mismatch {len(ref_pairs)} vs {len(result)}"
            )

            mismatches = 0
            for i, ref in enumerate(ref_pairs):
                _assert_pairs_match(ref, result.iloc[i], prefix, i)

    def test_trending_up(self):
        self._run_comparison(_make_trending_ohlcv(n=300, trend="up"))

    def test_trending_down(self):
        self._run_comparison(_make_trending_ohlcv(n=300, trend="down", seed=456))

    def test_volatile(self):
        self._run_comparison(_make_volatile_ohlcv(n=400))

    def test_short_series(self):
        """Edge case: very few bars, may have no confirmed levels."""
        self._run_comparison(_make_trending_ohlcv(n=50, trend="up"))


# ---------------------------------------------------------------------------
# Integration: full run_structure_feature_lab round-trip
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Edge case tests with hand-crafted data
# ---------------------------------------------------------------------------

def _make_base(n: int, start: str = "2022-01-01") -> pd.DataFrame:
    """Minimal base DataFrame for _build_fib_leg_features."""
    dates = pd.date_range(start, periods=n, freq="D", tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return pd.DataFrame({
        "close_time": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
        "atr": 1.0,
    })


def _make_ranked(levels: list[dict], defaults: dict | None = None) -> pd.DataFrame:
    """Build a ranked-levels DataFrame from explicit level dicts.

    Every dict must have: available_on, swing_date, value, level_score, level_scope.
    Optional: structure_label, swing_tier.
    """
    base = {
        "structure_label": None,
        "swing_tier": None,
    }
    if defaults:
        base.update(defaults)
    rows = [{**base, **lv} for lv in levels]
    df = pd.DataFrame(rows)
    for col in ("available_on", "swing_date"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


class TestEdgeCases:
    """Hand-crafted scenarios targeting specific divergence risks."""

    # -- 1. Tiebreaking: identical scores, different pair_available_on -------

    def test_tiebreaking_same_score(self):
        """Two pairs with identical base scores — the one with the later
        pair_available_on must win in both paths."""
        # Two highs with identical score, two lows with identical score
        ranked_highs = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-28",
             "value": 110.0, "level_score": 8.0, "level_scope": "major"},
            {"available_on": "2022-02-15", "swing_date": "2022-02-12",
             "value": 112.0, "level_score": 8.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-25",
             "value": 90.0, "level_score": 8.0, "level_scope": "major"},
            {"available_on": "2022-02-15", "swing_date": "2022-02-10",
             "value": 88.0, "level_score": 8.0, "level_scope": "major"},
        ])
        base = _make_base(120, start="2022-01-01")
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

    # -- 2. Cache path score accuracy over many days -------------------------

    def test_cache_path_score_does_not_drift(self):
        """When the pool is unchanged for 200+ consecutive days, the cached
        pair's score must still match the reference (no float drift)."""
        # One high, one low — pool never changes after they're both available
        ranked_highs = _make_ranked([
            {"available_on": "2022-01-10", "swing_date": "2022-01-08",
             "value": 105.0, "level_score": 10.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-01-10", "swing_date": "2022-01-05",
             "value": 95.0, "level_score": 10.0, "level_scope": "major"},
        ])
        # 300 days — the cache path will be used for ~290 of them
        base = _make_base(300, start="2022-01-01")
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

        # Extra: verify the last row's score is very precise (tight tolerance)
        last_ref = ref[-1]
        if last_ref:
            last_opt_score = result.iloc[-1][f"{prefix}_pair_score"]
            assert abs(last_opt_score - last_ref["pair_score_adjusted"]) < 1e-10, (
                f"Score drift after 300 days: ref={last_ref['pair_score_adjusted']:.15f} "
                f"opt={last_opt_score:.15f}"
            )

    # -- 3. Lookback boundary — level exactly at cutoff ----------------------

    def test_lookback_boundary_inclusion(self):
        """A level whose available_on falls exactly on the lookback cutoff
        boundary must be included/excluded identically in both paths."""
        # major scope: lookback_days=640
        # Place a level exactly 640 days before a test timestamp
        test_date = pd.Timestamp("2023-10-01", tz="UTC")
        boundary_date = test_date - pd.Timedelta(days=640)
        ranked_highs = _make_ranked([
            # Exactly at boundary — should be included (available_on >= cutoff)
            {"available_on": boundary_date, "swing_date": boundary_date - pd.Timedelta(days=2),
             "value": 110.0, "level_score": 8.0, "level_scope": "major"},
            # 1 day before boundary — should be excluded
            {"available_on": boundary_date - pd.Timedelta(days=1),
             "swing_date": boundary_date - pd.Timedelta(days=3),
             "value": 115.0, "level_score": 12.0, "level_scope": "major"},
            # Well within window
            {"available_on": test_date - pd.Timedelta(days=10),
             "swing_date": test_date - pd.Timedelta(days=12),
             "value": 108.0, "level_score": 7.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": test_date - pd.Timedelta(days=100),
             "swing_date": test_date - pd.Timedelta(days=102),
             "value": 90.0, "level_score": 8.0, "level_scope": "major"},
        ])
        # Build base that spans the boundary
        base = _make_base(700, start=str((boundary_date - pd.Timedelta(days=30)).date()))
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

    # -- 4. Pool transition — new level enters window -----------------------

    def test_pool_transition_new_level(self):
        """When a new level becomes available mid-series, the pair
        selection must update at exactly the same row in both paths."""
        ranked_highs = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-29",
             "value": 110.0, "level_score": 7.0, "level_scope": "major"},
            # Second high appears much later with a higher score
            {"available_on": "2022-05-01", "swing_date": "2022-04-28",
             "value": 120.0, "level_score": 12.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-01-15", "swing_date": "2022-01-12",
             "value": 90.0, "level_score": 8.0, "level_scope": "major"},
        ])
        base = _make_base(200, start="2022-01-01")
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        # Find the row where the pair changes
        pair_changed = False
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)
            if r and r.get("anchor_high") == 120.0:
                pair_changed = True
        assert pair_changed, "Expected the higher-value pair to activate"

    # -- 5. Global scope (no lookback) — pool only grows --------------------

    def test_global_no_lookback_pool_only_grows(self):
        """Global scope has lookback_days=None. The pool must only grow."""
        ranked_highs = _make_ranked([
            {"available_on": "2022-01-20", "swing_date": "2022-01-18",
             "value": 105.0, "level_score": 8.0, "level_scope": "global"},
            {"available_on": "2022-06-01", "swing_date": "2022-05-28",
             "value": 115.0, "level_score": 9.0, "level_scope": "global"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-01-10", "swing_date": "2022-01-08",
             "value": 92.0, "level_score": 8.0, "level_scope": "global"},
        ])
        base = _make_base(250, start="2022-01-01")
        scope = "global"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

    # -- 6. Single level each side ------------------------------------------

    def test_single_level_each_side(self):
        """Only one high and one low candidate. The simplest non-trivial case."""
        ranked_highs = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-30",
             "value": 105.0, "level_score": 7.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-25",
             "value": 95.0, "level_score": 7.0, "level_scope": "major"},
        ])
        base = _make_base(100, start="2022-01-01")
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

    # -- 7. All levels same score — recency tiebreaker ----------------------

    def test_all_same_score_recency_wins(self):
        """When all levels have the same level_score, the pair selection
        should be driven by recency penalties (age-based scoring).
        Both paths must agree."""
        ranked_highs = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-28",
             "value": 106.0, "level_score": 8.0, "level_scope": "major"},
            {"available_on": "2022-03-01", "swing_date": "2022-02-26",
             "value": 108.0, "level_score": 8.0, "level_scope": "major"},
            {"available_on": "2022-04-01", "swing_date": "2022-03-28",
             "value": 110.0, "level_score": 8.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-02-01", "swing_date": "2022-01-25",
             "value": 94.0, "level_score": 8.0, "level_scope": "major"},
            {"available_on": "2022-03-01", "swing_date": "2022-02-24",
             "value": 92.0, "level_score": 8.0, "level_scope": "major"},
            {"available_on": "2022-04-01", "swing_date": "2022-03-27",
             "value": 90.0, "level_score": 8.0, "level_scope": "major"},
        ])
        base = _make_base(200, start="2022-01-01")
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

    # -- 8. Empty pool then non-empty (late start) --------------------------

    def test_no_levels_then_levels_appear(self):
        """The first 50+ days have no available levels, then levels appear."""
        ranked_highs = _make_ranked([
            {"available_on": "2022-03-01", "swing_date": "2022-02-27",
             "value": 107.0, "level_score": 8.0, "level_scope": "major"},
        ])
        ranked_lows = _make_ranked([
            {"available_on": "2022-03-15", "swing_date": "2022-03-12",
             "value": 93.0, "level_score": 8.0, "level_scope": "major"},
        ])
        base = _make_base(150, start="2022-01-01")
        scope = "major"

        ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
        result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
        result = result.sort_values("close_time").reset_index(drop=True)

        prefix = f"{scope}_fib"
        # First ~73 rows should have no pair
        for i in range(73):
            pk = result.iloc[i].get(f"{prefix}_pair_key")
            assert pk is None or (isinstance(pk, float) and np.isnan(pk)), (
                f"Row {i}: expected no pair before levels available"
            )
        # All rows must match reference
        for i, r in enumerate(ref):
            _assert_pairs_match(r, result.iloc[i], prefix, i)

    # -- 9. Score precision: every row, tight tolerance ---------------------

    def test_score_precision_all_rows(self):
        """Compare score at 1e-10 tolerance for every single row on
        realistic-length synthetic data."""
        ohlcv = _make_volatile_ohlcv(n=400)
        structure, _ = simulate_btc_structure(ohlcv, BtcStructureConfig())
        ranked_highs = rank_confirmed_levels(structure.confirmed_highs, kind="high")
        ranked_lows = rank_confirmed_levels(structure.confirmed_lows, kind="low")

        base_cols = ["close_time", "open", "high", "low", "close", "volume", "atr"]
        base = structure.features[base_cols].copy()
        base["close_time"] = pd.to_datetime(base["close_time"], utc=True)

        for scope in ("local", "major", "global"):
            highs = _filter_fib_source_levels(ranked_highs, scope=scope)
            lows = _filter_fib_source_levels(ranked_lows, scope=scope)
            if highs.empty or lows.empty:
                continue

            ref = _reference_pair_selection(base, ranked_highs, ranked_lows, scope)
            result = _build_fib_leg_features(base.copy(), ranked_highs, ranked_lows, scope=scope)
            result = result.sort_values("close_time").reset_index(drop=True)
            prefix = f"{scope}_fib"

            for i, r in enumerate(ref):
                if not r:
                    continue
                opt_score = result.iloc[i][f"{prefix}_pair_score"]
                ref_score = r["pair_score_adjusted"]
                assert abs(opt_score - ref_score) < 1e-10, (
                    f"[{scope}] Row {i}: score precision failure: "
                    f"ref={ref_score:.15f} opt={opt_score:.15f} "
                    f"diff={abs(opt_score - ref_score):.2e}"
                )


class TestFullFeatureLabRoundTrip:
    """Verify the full feature lab produces consistent results."""

    def test_regime_columns_unchanged(self):
        ohlcv = _make_volatile_ohlcv(n=300)
        config = BtcStructureConfig()
        structure, _ = simulate_btc_structure(ohlcv, config)

        from btc_structure.features import STRUCTURE_REGIME
        lab = run_structure_feature_lab(structure, columns=STRUCTURE_REGIME)
        fm = lab.feature_matrix
        for col in STRUCTURE_REGIME:
            assert col in fm.columns, f"Missing column: {col}"
        assert len(fm) == len(ohlcv)

    def test_full_matrix_has_fib_columns(self):
        ohlcv = _make_volatile_ohlcv(n=200)
        config = BtcStructureConfig()
        structure, _ = simulate_btc_structure(ohlcv, config)
        lab = run_structure_feature_lab(structure)
        fm = lab.feature_matrix
        # Should have fib columns for each scope
        for scope in ("local", "major", "global"):
            assert f"{scope}_fib_pair_score" in fm.columns or fm.empty, (
                f"Missing {scope}_fib_pair_score"
            )
