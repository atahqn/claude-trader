from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from collections.abc import Iterator

from .models import Candle


def _normalize_time(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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


def interval_to_timedelta(interval: str) -> timedelta:
    return timedelta(seconds=interval_to_seconds(interval))


def floor_boundary(dt: datetime, interval: str) -> datetime:
    normalized = _normalize_time(dt)
    chunk_seconds = interval_to_seconds(interval)
    timestamp = int(normalized.timestamp())
    floored = timestamp - (timestamp % chunk_seconds)
    return datetime.fromtimestamp(floored, tz=UTC)


def next_boundary(dt: datetime, interval: str) -> datetime:
    return floor_boundary(dt, interval) + interval_to_timedelta(interval)


@dataclass(slots=True, frozen=True)
class PreviewSnapshot:
    candle: Candle
    source_period_start: datetime
    signal_time: datetime
    is_final: bool
    skipped_candle: Candle | None = None


@dataclass(slots=True)
class PartialCandleAccumulator:
    analysis_interval: str
    open_time: datetime | None = None
    close_time: datetime | None = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    taker_buy_volume: float = 0.0
    _analysis_duration: timedelta = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._analysis_duration = interval_to_timedelta(self.analysis_interval)

    def update(self, poll_bar: Candle) -> PreviewSnapshot:
        source_period_start = floor_boundary(poll_bar.open_time, self.analysis_interval)
        skipped_candle: Candle | None = None
        if self.open_time != source_period_start:
            if self.open_time is not None:
                skipped_candle = Candle(
                    open_time=self.open_time,
                    close_time=self.close_time,
                    open=self.open,
                    high=self.high,
                    low=self.low,
                    close=self.close,
                    volume=self.volume,
                    taker_buy_volume=self.taker_buy_volume,
                )
            self.open_time = source_period_start
            self.close_time = poll_bar.close_time
            self.open = poll_bar.open
            self.high = poll_bar.high
            self.low = poll_bar.low
            self.close = poll_bar.close
            self.volume = poll_bar.volume
            self.taker_buy_volume = poll_bar.taker_buy_volume
        else:
            self.close_time = poll_bar.close_time
            self.high = max(self.high, poll_bar.high)
            self.low = min(self.low, poll_bar.low)
            self.close = poll_bar.close
            self.volume += poll_bar.volume
            self.taker_buy_volume += poll_bar.taker_buy_volume

        candle = Candle(
            open_time=self.open_time,
            close_time=self.close_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            taker_buy_volume=self.taker_buy_volume,
        )
        is_final = candle.close_time >= source_period_start + self._analysis_duration
        return PreviewSnapshot(
            candle=candle,
            source_period_start=source_period_start,
            signal_time=candle.close_time,
            is_final=is_final,
            skipped_candle=skipped_candle,
        )

    def reset(self) -> None:
        self.open_time = None
        self.close_time = None
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.volume = 0.0
        self.taker_buy_volume = 0.0


def iter_preview_snapshots(
    poll_candles: list[Candle],
    analysis_interval: str,
) -> Iterator[PreviewSnapshot]:
    accumulator = PartialCandleAccumulator(analysis_interval)
    for poll_bar in poll_candles:
        snapshot = accumulator.update(poll_bar)
        yield snapshot
        if snapshot.is_final:
            accumulator.reset()


@dataclass(slots=True)
class SourcePeriodGate:
    _last_emitted: dict[object, datetime] = field(default_factory=dict)

    def allow(self, source_period_start: datetime, *, key: object = "default") -> bool:
        return self._last_emitted.get(key) != source_period_start

    def mark(self, source_period_start: datetime, *, key: object = "default") -> None:
        self._last_emitted[key] = source_period_start

    def clear(self, *, key: object | None = None) -> None:
        if key is None:
            self._last_emitted.clear()
            return
        self._last_emitted.pop(key, None)
