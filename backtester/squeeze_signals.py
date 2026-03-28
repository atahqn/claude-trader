from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import MarketType, PositionType, Signal
from .preview import SourcePeriodGate


def build_squeeze_signal(
    *,
    signal_date: datetime,
    position_type: PositionType,
    symbol: str,
    config: Any,
    metadata: dict[str, object],
) -> Signal:
    max_holding_hours = getattr(config, "max_holding_hours", None)
    return Signal(
        signal_date=signal_date,
        position_type=position_type,
        ticker=symbol,
        tp_pct=config.short_tp if position_type is PositionType.SHORT else config.long_tp,
        sl_pct=config.short_sl if position_type is PositionType.SHORT else config.long_sl,
        leverage=config.leverage,
        market_type=getattr(config, "market_type", MarketType.FUTURES),
        taker_fee_rate=getattr(config, "taker_fee_rate", 0.0005),
        entry_delay_seconds=5,
        max_holding_hours=max_holding_hours,
        metadata=metadata,
    )


def emit_squeeze_entry_signals(
    *,
    signal_date: datetime,
    symbol: str,
    config: Any,
    strategy_name: str,
    prev_squeeze_count: int,
    squeeze_on: bool,
    mom: float,
    rsi: float,
    atr_ratio: float,
    ret_72h: float,
    last_short: datetime | None,
    last_long: datetime | None,
    source_period_start: datetime | None = None,
    source_period_gate: SourcePeriodGate | None = None,
    short_gate_key: object = "default",
    long_gate_key: object = "default",
) -> tuple[list[Signal], datetime | None, datetime | None]:
    if atr_ratio > config.atr_ratio_max:
        return [], last_short, last_long
    if prev_squeeze_count < config.min_squeeze_bars or squeeze_on:
        return [], last_short, last_long

    signals: list[Signal] = []
    preview_metadata: dict[str, object] = {}
    if source_period_start is not None:
        preview_metadata = {
            "analysis_interval": getattr(config, "analysis_interval", "1h"),
            "poll_interval": getattr(config, "effective_poll_interval", getattr(config, "analysis_interval", "1h")),
            "source_hour_start": source_period_start.isoformat(),
        }

    enable_short = getattr(config, "enable_short", True)
    enable_long = getattr(config, "enable_long", True)

    short_gate_ok = (
        source_period_start is None
        or source_period_gate is None
        or source_period_gate.allow(source_period_start, key=short_gate_key)
    )
    if (
        enable_short
        and short_gate_ok
        and mom < 0
        and rsi >= config.short_rsi_floor
        and (
            last_short is None
            or (signal_date - last_short).total_seconds() >= config.short_cooldown_h * 3600
        )
    ):
        last_short = signal_date
        if source_period_start is not None and source_period_gate is not None:
            source_period_gate.mark(source_period_start, key=short_gate_key)
        signals.append(
            build_squeeze_signal(
                signal_date=signal_date,
                position_type=PositionType.SHORT,
                symbol=symbol,
                config=config,
                metadata={
                    "strategy": f"{strategy_name}_short",
                    **preview_metadata,
                    "mom": round(mom, 6),
                    "rsi": round(rsi, 1),
                    "atr_ratio": round(atr_ratio, 2),
                    "sq_count": int(prev_squeeze_count),
                },
            )
        )

    long_gate_ok = (
        source_period_start is None
        or source_period_gate is None
        or source_period_gate.allow(source_period_start, key=long_gate_key)
    )
    if (
        enable_long
        and long_gate_ok
        and mom > 0
        and rsi <= config.long_rsi_cap
        and ret_72h >= config.long_regime_min
        and (
            last_long is None
            or (signal_date - last_long).total_seconds() >= config.long_cooldown_h * 3600
        )
    ):
        last_long = signal_date
        if source_period_start is not None and source_period_gate is not None:
            source_period_gate.mark(source_period_start, key=long_gate_key)
        signals.append(
            build_squeeze_signal(
                signal_date=signal_date,
                position_type=PositionType.LONG,
                symbol=symbol,
                config=config,
                metadata={
                    "strategy": f"{strategy_name}_long",
                    **preview_metadata,
                    "mom": round(mom, 6),
                    "rsi": round(rsi, 1),
                    "atr_ratio": round(atr_ratio, 2),
                    "ret_72h": round(ret_72h, 1),
                    "sq_count": int(prev_squeeze_count),
                },
            )
        )

    return signals, last_short, last_long
