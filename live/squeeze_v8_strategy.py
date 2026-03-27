"""Live signal generator: Squeeze V8 Strategy (SHORT + LONG).

V8 widens SHORT TP/SL from 2.0/1.0 to 3.0/1.5, lowers RSI floor from 30 to 25,
and relaxes ATR ratio gate from 1.3 to 1.5. LONG side unchanged from V7.

Backtested across 5 holdout periods (Oct 2023 – Mar 2026, ~55 weeks total):
  V8: +79.80% last 3 months, +64.39% on two older holdout periods
  vs V7: +33.80% / +32.17% on the same windows
  V8 doubles V7 PNL with 59.3% trade win rate (vs 50.7%).

Architecture:
  Two sub-signals from the SAME structural event (BB/KC squeeze release):

  1. SQUEEZE SHORT (V8: wider TP/SL, lower RSI floor):
    - 7+ bar compression, release with negative momentum
    - RSI >= 25 (not deeply oversold), ATR ratio <= 1.5
    - TP: 3.0%, SL: 1.5%, Cooldown: 12h
    - Changed from V7: TP 2.0→3.0, SL 1.0→1.5, RSI 30→25, ATR 1.3→1.5

  2. SQUEEZE LONG (unchanged from V7):
    - 7+ bar compression, release with POSITIVE momentum
    - ret_72h >= 6% (STRONG BULL regime only)
    - RSI <= 70 (not overbought), ATR ratio <= 1.5
    - TP: 4.0%, SL: 2.0%, Cooldown: 12h

Polling:
  - analysis interval stays 1h
  - poll interval defaults to 1h, preserving the old close-only behavior
  - when poll interval is lower (for example 15m or 5m), the strategy evaluates
    Binance's current in-progress 1h candle snapshot at those poll boundaries
  - uses incremental _SqueezeV7PreviewState per symbol for O(1) indicator updates

Look-ahead bias prevention:
  - close-only mode uses the last fully closed 1h candle
  - preview mode uses the currently visible 1h candle snapshot from Binance
  - all indicators remain backward-looking: rolling(), ewm(), pct_change()
  - entry executes via market order after signal
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import numpy as np

from backtester.indicators import required_warmup
from backtester.models import Candle, Signal
from backtester.preview import SourcePeriodGate, floor_boundary, interval_to_seconds
from backtester.squeeze_signals import emit_squeeze_entry_signals
from backtester.strategies.squeeze_v7 import (
    SQUEEZE_V7_FEATURE_COLUMNS,
    _SqueezeV7PreviewState,
    _safe_feature_value,
)

from .auth_client import LiveMarketClient
from .signal_generator import SignalGenerator

SYMBOLS = [
    "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
    "ENA/USDT", "INJ/USDT", "NEAR/USDT", "ALGO/USDT",
    "RENDER/USDT", "WIF/USDT", "ADA/USDT", "APT/USDT",
]

_LOOKBACK_BARS = 100
_WARMUP_BARS = required_warmup(SQUEEZE_V7_FEATURE_COLUMNS)
_SUPPORTED_POLL_INTERVALS = {"1h", "30m", "15m", "5m", "1m"}


class SqueezeV8Strategy(SignalGenerator):
    """Squeeze V8: SHORT + LONG live signal generator.

    SHORT: squeeze breakout with negative momentum (all regimes)
           TP 3.0/SL 1.5, RSI >= 25, ATR ratio <= 1.5
    LONG:  squeeze breakout with positive momentum (bull regime only, ret_72h >= 6%)
           TP 4.0/SL 2.0, RSI <= 70, ATR ratio <= 1.5
    """

    def __init__(
        self,
        analysis_interval: str = "1h",
        poll_interval: str | None = None,
        leverage: float = 1.0,
        # SHORT params (V8: wider TP/SL, lower RSI floor)
        short_tp: float = 3.0,
        short_sl: float = 1.5,
        short_cooldown_h: float = 12.0,
        short_rsi_floor: float = 25.0,
        # LONG params (unchanged from V7)
        long_tp: float = 4.0,
        long_sl: float = 2.0,
        long_cooldown_h: float = 12.0,
        long_rsi_cap: float = 70.0,
        long_regime_min: float = 6.0,
        # Shared params (V8: relaxed ATR gate)
        min_squeeze_bars: int = 7,
        atr_ratio_max: float = 1.5,
    ) -> None:
        if analysis_interval != "1h":
            raise ValueError("SqueezeV8 live strategy currently supports only 1h analysis_interval")
        effective_poll_interval = poll_interval or analysis_interval
        if effective_poll_interval not in _SUPPORTED_POLL_INTERVALS:
            raise ValueError("poll_interval must be one of 1h, 30m, 15m, 5m, 1m")
        if interval_to_seconds(analysis_interval) % interval_to_seconds(effective_poll_interval) != 0:
            raise ValueError("poll_interval must divide analysis_interval exactly")

        self.analysis_interval = analysis_interval
        self.poll_interval = poll_interval
        self.enable_short = True
        self.enable_long = True
        self.leverage = leverage
        self.short_tp = short_tp
        self.short_sl = short_sl
        self.short_cooldown_h = short_cooldown_h
        self.short_rsi_floor = short_rsi_floor
        self.long_tp = long_tp
        self.long_sl = long_sl
        self.long_cooldown_h = long_cooldown_h
        self.long_rsi_cap = long_rsi_cap
        self.long_regime_min = long_regime_min
        self.min_squeeze_bars = min_squeeze_bars
        self.atr_ratio_max = atr_ratio_max

        self._client: LiveMarketClient | None = None
        self._last_short: dict[str, datetime] = {}
        self._last_long: dict[str, datetime] = {}
        self._source_period_gate = SourcePeriodGate()
        self._states: dict[str, _SqueezeV7PreviewState] = {}
        self._last_committed_hour: dict[str, datetime] = {}

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        self._warm_up_states()
        print(
            f"SqueezeV8 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"analysis={self.analysis_interval} | poll={self.effective_poll_interval} | "
            f"SHORT TP/SL={self.short_tp}/{self.short_sl}% "
            f"CD={self.short_cooldown_h}h RSI>={self.short_rsi_floor} | "
            f"LONG TP/SL={self.long_tp}/{self.long_sl}% "
            f"CD={self.long_cooldown_h}h RSI<={self.long_rsi_cap} "
            f"regime>={self.long_regime_min}% | "
            f"min_sq={self.min_squeeze_bars} ATR<={self.atr_ratio_max}",
            file=sys.stderr,
        )

    def _warm_up_states(self) -> None:
        assert self._client is not None
        now = datetime.now(UTC)
        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        for symbol in SYMBOLS:
            try:
                candles = self._client.fetch_klines(
                    symbol=symbol.replace("/", ""),
                    interval=self.analysis_interval,
                    start=start,
                    end=now,
                )
                state = _SqueezeV7PreviewState()
                for candle in candles:
                    if candle.close_time <= now:
                        state.commit(candle)
                        self._last_committed_hour[symbol] = candle.open_time
                self._states[symbol] = state
                print(
                    f"  {symbol}: warmed up {state.candle_count} bars",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"  {symbol}: warmup failed: {exc}", file=sys.stderr)
                self._states[symbol] = _SqueezeV7PreviewState()

    def poll(self) -> list[Signal] | None:
        assert self._client is not None
        now = self.current_time()
        signals: list[Signal] = []

        for symbol in SYMBOLS:
            try:
                sigs = self._check_symbol(symbol, now)
                signals.extend(sigs)
            except Exception as exc:
                print(f"Error checking {symbol}: {exc}", file=sys.stderr)

        return signals if signals else None

    def _check_symbol(self, symbol: str, now: datetime) -> list[Signal]:
        assert self._client is not None

        state = self._states.get(symbol)
        if state is None:
            return []

        preview_enabled = self.effective_poll_interval != self.analysis_interval

        # Commit any newly closed hourly candles since last commit
        last_committed = self._last_committed_hour.get(symbol)
        current_hour = floor_boundary(now, self.analysis_interval)
        if last_committed is None or last_committed < current_hour:
            new_candles = self._client.fetch_klines(
                symbol=symbol.replace("/", ""),
                interval=self.analysis_interval,
                start=(last_committed or current_hour - timedelta(hours=2)),
                end=now,
            )
            for candle in new_candles:
                candle_hour = candle.open_time
                if candle.close_time <= now and (
                    last_committed is None or candle_hour > last_committed
                ):
                    state.commit(candle)
                    self._last_committed_hour[symbol] = candle_hour

        if state.last_row is None:
            return []
        if (
            state.candle_count < _WARMUP_BARS
            and (np.isnan(state.last_row.mom_slope) or np.isnan(state.last_row.atr_ratio))
        ):
            return []

        # Get the current candle to evaluate
        if preview_enabled:
            # Fetch the current open 1h candle snapshot from Binance
            candles = self._client.fetch_klines(
                symbol=symbol.replace("/", ""),
                interval=self.analysis_interval,
                start=current_hour,
                end=now + timedelta(hours=1),
            )
            if not candles:
                return []
            eval_candle = candles[-1]
            if eval_candle.close_time <= now:
                # Hour already closed, was committed above — nothing to preview
                return []
        else:
            # Non-preview: evaluate the last committed candle
            if state.last_row is None:
                return []
            eval_candle = Candle(
                open_time=state.last_row.open_time,
                close_time=state.last_row.close_time,
                open=0.0,
                high=0.0,
                low=0.0,
                close=state.prev_close or 0.0,
                volume=0.0,
            )
            # For non-preview, use the committed last_row directly
            row = state.last_row
            prev_row = state.last_row  # Need the row before last_row
            # Non-preview path: we need the previous bar's squeeze_count.
            # Since we only have the last committed row, we use it as "current"
            # and need the one before it. Fall back to pandas for non-preview.
            return self._check_symbol_pandas(symbol, now)

        # Preview path: evaluate partial candle with incremental state
        step = state.preview(eval_candle)
        row = step.row
        prev = state.last_row

        if np.isnan(row.mom_slope) or np.isnan(row.atr_ratio):
            return []

        atr_ratio = _safe_feature_value(row.atr_ratio, 1.0)
        if atr_ratio > self.atr_ratio_max:
            return []

        if prev.squeeze_count < self.min_squeeze_bars or row.squeeze_on:
            return []

        rsi = _safe_feature_value(row.rsi_14, 50.0)
        ret_72h = _safe_feature_value(row.ret_72h, 0.0)
        mom = _safe_feature_value(row.mom_slope, 0.0)

        source_hour_start = eval_candle.open_time
        emitted, self._last_short[symbol], self._last_long[symbol] = emit_squeeze_entry_signals(
            signal_date=now,
            symbol=symbol,
            config=self,
            strategy_name="squeeze_v8",
            prev_squeeze_count=int(prev.squeeze_count),
            squeeze_on=row.squeeze_on,
            mom=mom,
            rsi=rsi,
            atr_ratio=atr_ratio,
            ret_72h=ret_72h,
            last_short=self._last_short.get(symbol),
            last_long=self._last_long.get(symbol),
            source_period_start=source_hour_start,
            source_period_gate=self._source_period_gate,
            short_gate_key=symbol,
            long_gate_key=symbol,
        )
        return emitted

    def _check_symbol_pandas(self, symbol: str, now: datetime) -> list[Signal]:
        """Fallback for non-preview (hourly) polling: full pandas recompute.

        Only used when poll_interval == analysis_interval (default 1h mode).
        """
        assert self._client is not None
        import pandas as pd
        from backtester.strategies.squeeze_v7 import build_squeeze_v7_feature_frame

        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval=self.analysis_interval,
            start=start,
            end=now,
        )

        if len(candles) < _WARMUP_BARS:
            return []

        rows = []
        for c in candles:
            rows.append({
                "open_time": c.open_time,
                "close_time": c.close_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return []
        df = df.sort_values("open_time").reset_index(drop=True)
        df = build_squeeze_v7_feature_frame(df)

        last_idx = len(df) - 1
        if df.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx + 1 < _WARMUP_BARS:
            return []

        row = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]

        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            return []

        rsi = row.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50
        atr_ratio = row.get("atr_ratio", 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0
        ret_72h = row.get("ret_72h", 0)
        if pd.isna(ret_72h):
            ret_72h = 0

        if atr_ratio > self.atr_ratio_max:
            return []
        if prev["squeeze_count"] < self.min_squeeze_bars or row["squeeze_on"]:
            return []

        mom = row["mom_slope"]
        emitted, self._last_short[symbol], self._last_long[symbol] = emit_squeeze_entry_signals(
            signal_date=now,
            symbol=symbol,
            config=self,
            strategy_name="squeeze_v8",
            prev_squeeze_count=int(prev["squeeze_count"]),
            squeeze_on=bool(row["squeeze_on"]),
            mom=float(mom),
            rsi=float(rsi),
            atr_ratio=float(atr_ratio),
            ret_72h=float(ret_72h),
            last_short=self._last_short.get(symbol),
            last_long=self._last_long.get(symbol),
        )
        return emitted
