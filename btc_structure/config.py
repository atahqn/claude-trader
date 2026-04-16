from __future__ import annotations

from dataclasses import dataclass

BASE_LEVEL_DAYS = (1, 3, 7, 10, 30, 90, 180, 300, 365)
FIB_BASE_RATIOS = (0.0, 0.236, 0.34, 0.382, 0.5, 0.618, 0.66, 0.786, 1.0)
FIB_EXTENSION_RATIOS = (1.272, 1.618, 2.0)


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


@dataclass(slots=True, frozen=True)
class BtcStructureConfig:
    interval: str = "1d"
    market_type: str = "futures"
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
        market_type: str = "futures",
        years: int = 5,
    ) -> BtcStructureConfig:
        seconds = interval_to_seconds(interval)
        bars_per_day = max(1, int(round(86400 / seconds)))

        def bars(days: int) -> int:
            return max(1, int(round(days * bars_per_day)))

        return BtcStructureConfig(
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
