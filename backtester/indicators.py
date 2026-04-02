from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

_RSI_ALPHA = 1 / 14
_SLOPE_XS_20 = np.arange(20, dtype=float)
_SLOPE_XS_20_SUM = float(_SLOPE_XS_20.sum())
_SLOPE_XS_20_DOT = float(np.dot(_SLOPE_XS_20, _SLOPE_XS_20))
_SLOPE_DENOM_20 = 20 * _SLOPE_XS_20_DOT - _SLOPE_XS_20_SUM ** 2
_RAW_INDICATOR_INPUTS = frozenset({"open", "high", "low", "close", "volume"})


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    name: str
    dependencies: tuple[str, ...]
    additional_bars: int


def compute_rsi_from_ewm_means(gain_mean: pd.Series | float, loss_mean: pd.Series | float) -> pd.Series | float:
    if isinstance(gain_mean, pd.Series) or isinstance(loss_mean, pd.Series):
        index = gain_mean.index if isinstance(gain_mean, pd.Series) else loss_mean.index
        gain_series = gain_mean if isinstance(gain_mean, pd.Series) else pd.Series(gain_mean, index=index)
        loss_series = loss_mean if isinstance(loss_mean, pd.Series) else pd.Series(loss_mean, index=index)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = gain_series / loss_series
            rsi = 100.0 - (100.0 / (1.0 + rs))
        zero_loss = loss_series == 0.0
        zero_gain = gain_series == 0.0
        rsi = rsi.where(~zero_loss, 100.0)
        return rsi.where(~(zero_loss & zero_gain), np.nan)
    if np.isnan(gain_mean) or np.isnan(loss_mean):
        return np.nan
    if loss_mean == 0.0:
        if gain_mean == 0.0:
            return np.nan
        return 100.0
    rs = gain_mean / loss_mean
    return 100.0 - (100.0 / (1.0 + rs))


def true_range_series(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def true_range_value(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None:
        return high - low
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def linear_regression_slope(values: Sequence[float] | np.ndarray) -> float:
    count = len(values)
    if count == 20:
        ys = np.asarray(values, dtype=float)
        numerator = 20 * float(np.dot(_SLOPE_XS_20, ys)) - _SLOPE_XS_20_SUM * float(ys.sum())
        if _SLOPE_DENOM_20 == 0.0:
            return 0.0
        return numerator / _SLOPE_DENOM_20
    xs = np.arange(count, dtype=float)
    ys = np.asarray(values, dtype=float)
    numerator = count * float(np.dot(xs, ys)) - float(xs.sum()) * float(ys.sum())
    denominator = count * float(np.dot(xs, xs)) - float(xs.sum()) ** 2
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _rolling_linear_regression_slope(values: np.ndarray) -> float:
    return linear_regression_slope(values)


def _compute_squeeze_count(squeeze_on: pd.Series) -> pd.Series:
    counts: list[int] = []
    count = 0
    for active in squeeze_on:
        count = count + 1 if bool(active) else 0
        counts.append(count)
    return pd.Series(counts, index=squeeze_on.index, dtype="int64")


_INDICATOR_SPECS: dict[str, IndicatorSpec] = {
    "_delta_close": IndicatorSpec("_delta_close", ("close",), 1),
    "_gain_ewm_14": IndicatorSpec("_gain_ewm_14", ("_delta_close",), 13),
    "_loss_ewm_14": IndicatorSpec("_loss_ewm_14", ("_delta_close",), 13),
    "rsi_14": IndicatorSpec("rsi_14", ("_gain_ewm_14", "_loss_ewm_14"), 0),
    "true_range": IndicatorSpec("true_range", ("high", "low", "close"), 0),
    "atr_14": IndicatorSpec("atr_14", ("true_range",), 13),
    "atr_72_avg": IndicatorSpec("atr_72_avg", ("atr_14",), 71),
    "atr_ratio": IndicatorSpec("atr_ratio", ("atr_14", "atr_72_avg"), 0),
    "ret_24h": IndicatorSpec("ret_24h", ("close",), 24),
    "ret_48h": IndicatorSpec("ret_48h", ("close",), 48),
    "ret_72h": IndicatorSpec("ret_72h", ("close",), 72),
    "vol_sma_20": IndicatorSpec("vol_sma_20", ("volume",), 19),
    "vol_ratio": IndicatorSpec("vol_ratio", ("volume", "vol_sma_20"), 0),
    "_bb_ma_20": IndicatorSpec("_bb_ma_20", ("close",), 19),
    "_bb_std_20": IndicatorSpec("_bb_std_20", ("close",), 19),
    "bb_upper": IndicatorSpec("bb_upper", ("_bb_ma_20", "_bb_std_20"), 0),
    "bb_lower": IndicatorSpec("bb_lower", ("_bb_ma_20", "_bb_std_20"), 0),
    "ema_20": IndicatorSpec("ema_20", ("close",), 0),
    "kc_upper": IndicatorSpec("kc_upper", ("ema_20", "atr_14"), 0),
    "kc_lower": IndicatorSpec("kc_lower", ("ema_20", "atr_14"), 0),
    "squeeze_on": IndicatorSpec("squeeze_on", ("bb_upper", "bb_lower", "kc_upper", "kc_lower"), 0),
    "squeeze_count": IndicatorSpec("squeeze_count", ("squeeze_on",), 0),
    "mom_slope": IndicatorSpec("mom_slope", ("close",), 19),
    "body": IndicatorSpec("body", ("open", "close"), 0),
    "body_ratio": IndicatorSpec("body_ratio", ("body", "high", "low"), 0),
    # --- ADX (Average Directional Index) ---
    "_plus_dm": IndicatorSpec("_plus_dm", ("high", "low"), 1),
    "_minus_dm": IndicatorSpec("_minus_dm", ("high", "low"), 1),
    "_smoothed_plus_dm": IndicatorSpec("_smoothed_plus_dm", ("_plus_dm",), 13),
    "_smoothed_minus_dm": IndicatorSpec("_smoothed_minus_dm", ("_minus_dm",), 13),
    "_smoothed_tr": IndicatorSpec("_smoothed_tr", ("true_range",), 13),
    "_plus_di": IndicatorSpec("_plus_di", ("_smoothed_plus_dm", "_smoothed_tr"), 0),
    "_minus_di": IndicatorSpec("_minus_di", ("_smoothed_minus_dm", "_smoothed_tr"), 0),
    "_dx": IndicatorSpec("_dx", ("_plus_di", "_minus_di"), 0),
    "adx_14": IndicatorSpec("adx_14", ("_dx",), 13),
    # --- Bollinger Band normalizations ---
    "bb_pct_b": IndicatorSpec("bb_pct_b", ("close", "bb_upper", "bb_lower"), 0),
    "bb_width": IndicatorSpec("bb_width", ("bb_upper", "bb_lower", "_bb_ma_20"), 0),
}


def _compute_indicator(name: str, frame: pd.DataFrame) -> pd.Series:
    if name == "_delta_close":
        return frame["close"].diff()
    if name == "_gain_ewm_14":
        return frame["_delta_close"].clip(lower=0).ewm(alpha=_RSI_ALPHA, min_periods=14).mean()
    if name == "_loss_ewm_14":
        return (-frame["_delta_close"]).clip(lower=0).ewm(alpha=_RSI_ALPHA, min_periods=14).mean()
    if name == "rsi_14":
        return compute_rsi_from_ewm_means(frame["_gain_ewm_14"], frame["_loss_ewm_14"])
    if name == "true_range":
        return true_range_series(frame)
    if name == "atr_14":
        return frame["true_range"].rolling(14).mean()
    if name == "atr_72_avg":
        return frame["atr_14"].rolling(72).mean()
    if name == "atr_ratio":
        return frame["atr_14"] / frame["atr_72_avg"]
    if name == "ret_24h":
        return frame["close"].pct_change(24) * 100.0
    if name == "ret_48h":
        return frame["close"].pct_change(48) * 100.0
    if name == "ret_72h":
        return frame["close"].pct_change(72) * 100.0
    if name == "vol_sma_20":
        return frame["volume"].rolling(20).mean()
    if name == "vol_ratio":
        return frame["volume"] / frame["vol_sma_20"]
    if name == "_bb_ma_20":
        return frame["close"].rolling(20).mean()
    if name == "_bb_std_20":
        return frame["close"].rolling(20).std()
    if name == "bb_upper":
        return frame["_bb_ma_20"] + 2.0 * frame["_bb_std_20"]
    if name == "bb_lower":
        return frame["_bb_ma_20"] - 2.0 * frame["_bb_std_20"]
    if name == "ema_20":
        return frame["close"].ewm(span=20).mean()
    if name == "kc_upper":
        return frame["ema_20"] + 1.5 * frame["atr_14"]
    if name == "kc_lower":
        return frame["ema_20"] - 1.5 * frame["atr_14"]
    if name == "squeeze_on":
        return (frame["bb_lower"] > frame["kc_lower"]) & (frame["bb_upper"] < frame["kc_upper"])
    if name == "squeeze_count":
        return _compute_squeeze_count(frame["squeeze_on"])
    if name == "mom_slope":
        return frame["close"].rolling(20).apply(_rolling_linear_regression_slope, raw=True)
    if name == "body":
        return frame["close"] - frame["open"]
    if name == "body_ratio":
        return frame["body"] / (frame["high"] - frame["low"]).replace(0, np.nan)
    # --- ADX ---
    if name == "_plus_dm":
        diff_high = frame["high"].diff()
        diff_low = -(frame["low"].diff())
        return diff_high.where((diff_high > diff_low) & (diff_high > 0), 0.0)
    if name == "_minus_dm":
        diff_high = frame["high"].diff()
        diff_low = -(frame["low"].diff())
        return diff_low.where((diff_low > diff_high) & (diff_low > 0), 0.0)
    if name == "_smoothed_plus_dm":
        return frame["_plus_dm"].ewm(alpha=_RSI_ALPHA, min_periods=14).mean()
    if name == "_smoothed_minus_dm":
        return frame["_minus_dm"].ewm(alpha=_RSI_ALPHA, min_periods=14).mean()
    if name == "_smoothed_tr":
        return frame["true_range"].ewm(alpha=_RSI_ALPHA, min_periods=14).mean()
    if name == "_plus_di":
        return 100.0 * frame["_smoothed_plus_dm"] / frame["_smoothed_tr"]
    if name == "_minus_di":
        return 100.0 * frame["_smoothed_minus_dm"] / frame["_smoothed_tr"]
    if name == "_dx":
        di_sum = frame["_plus_di"] + frame["_minus_di"]
        di_diff = (frame["_plus_di"] - frame["_minus_di"]).abs()
        return (100.0 * di_diff / di_sum).replace([np.inf, -np.inf], np.nan)
    if name == "adx_14":
        return frame["_dx"].ewm(alpha=_RSI_ALPHA, min_periods=14).mean()
    # --- BB normalizations ---
    if name == "bb_pct_b":
        band_range = frame["bb_upper"] - frame["bb_lower"]
        return ((frame["close"] - frame["bb_lower"]) / band_range).replace([np.inf, -np.inf], np.nan)
    if name == "bb_width":
        return (frame["bb_upper"] - frame["bb_lower"]) / frame["_bb_ma_20"]
    raise KeyError(f"unknown indicator: {name}")


def _resolve_indicator_order(indicators: Sequence[str]) -> list[str]:
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in _RAW_INDICATOR_INPUTS or name in visited:
            return
        if name in visiting:
            raise ValueError(f"cyclic indicator dependency detected at {name}")
        spec = _INDICATOR_SPECS.get(name)
        if spec is None:
            raise KeyError(f"unknown indicator: {name}")
        visiting.add(name)
        for dependency in spec.dependencies:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for indicator in indicators:
        visit(indicator)
    return order


@lru_cache(maxsize=None)
def _required_warmup_single(name: str) -> int:
    if name in _RAW_INDICATOR_INPUTS:
        return 1
    spec = _INDICATOR_SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown indicator: {name}")
    dependency_bars = max((_required_warmup_single(dep) for dep in spec.dependencies), default=1)
    return dependency_bars + spec.additional_bars


def required_warmup(indicators: Sequence[str]) -> int:
    if not indicators:
        return 0
    return max(_required_warmup_single(name) for name in indicators)


def compute_indicator_frame(frame: pd.DataFrame, indicators: Sequence[str]) -> pd.DataFrame:
    if frame.empty or not indicators:
        return frame.copy()

    requested = tuple(dict.fromkeys(indicators))
    order = _resolve_indicator_order(requested)
    original_columns = set(frame.columns)
    df = frame.copy()

    for name in order:
        df[name] = _compute_indicator(name, df)

    internal_columns = [
        name
        for name in order
        if name not in requested and name not in original_columns
    ]
    if internal_columns:
        df = df.drop(columns=internal_columns)
    return df
