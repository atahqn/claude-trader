"""Live signal generator: Squeeze V6B Strategy (SHORT + LONG + DIP_TR).

Extends V6 by adding the `DIP_TR` long signal from `TRIP_R6T30`.

Architecture:
  Three sub-signals:

  1. SQUEEZE SHORT (unchanged from V5/V6):
    - 7+ bar compression, release with negative momentum
    - RSI >= 30, ATR ratio <= 1.3
    - TP: 2.0%, SL: 1.0%, Cooldown: 12h

  2. SQUEEZE LONG (V6):
    - 7+ bar compression, release with positive momentum
    - ret_72h >= 6%, RSI <= 70, ATR ratio <= 1.3
    - TP: 3.0%, SL: 1.5%, Cooldown: 12h

  3. DIP_TR (new in V6B):
    - ret_24h <= -3.0% pullback inside a non-bear regime
    - body_ratio >= 0.2 and vol_ratio >= 1.2 on the bounce candle
    - 25 <= RSI <= 50, ret_72h >= 0, ATR ratio <= 1.5
    - TP: 2.0%, SL: 1.0%, Cooldown: 24h

Validation target from the Round 3 research script:
  TRIP_R6T30:
    - Weekly win rate: 87% (27/31)
    - Total PNL: +103.29%
    - Worst week: -8.00%
    - Trades: 240

Look-ahead bias prevention:
  - Uses the most recent completed hourly candle only
  - All indicators are backward-looking
  - Signals execute via market orders after the poll
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backtester.models import MarketType, PositionType, Signal

from .auth_client import LiveMarketClient
from .signal_generator import SignalGenerator
from .squeeze_v6_strategy import SYMBOLS, _LOOKBACK_BARS, _WARMUP_BARS, _candles_to_df

TAKER_FEE = 0.0005


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the indicator set needed by TRIP_R6T30."""
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_72_avg"] = df["atr_14"].rolling(72).mean()
    df["atr_ratio"] = df["atr_14"] / df["atr_72_avg"]

    df["ret_72h"] = c.pct_change(72) * 100
    df["ret_24h"] = c.pct_change(24) * 100

    df["vol_sma_20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_sma_20"]

    bb_ma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_ma + 2 * bb_std
    df["bb_lower"] = bb_ma - 2 * bb_std

    ema_20 = c.ewm(span=20).mean()
    df["kc_upper"] = ema_20 + 1.5 * df["atr_14"]
    df["kc_lower"] = ema_20 - 1.5 * df["atr_14"]

    df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    cnt = 0
    counts = []
    for s in df["squeeze_on"]:
        cnt = cnt + 1 if s else 0
        counts.append(cnt)
    df["squeeze_count"] = counts

    df["mom_slope"] = c.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    rng = h - l
    df["body"] = c - o
    df["body_ratio"] = df["body"] / rng.replace(0, np.nan)

    return df


class SqueezeV6BStrategy(SignalGenerator):
    """TRIP_R6T30 live strategy: squeeze SHORT + squeeze LONG + DIP_TR."""

    def __init__(
        self,
        leverage: float = 1.0,
        short_tp: float = 2.0,
        short_sl: float = 1.0,
        short_cooldown_h: float = 12.0,
        short_rsi_floor: float = 30.0,
        short_atr_max: float = 1.3,
        long_tp: float = 3.0,
        long_sl: float = 1.5,
        long_cooldown_h: float = 12.0,
        long_rsi_cap: float = 70.0,
        long_regime_min: float = 6.0,
        long_atr_max: float = 1.3,
        dip_tp: float = 2.0,
        dip_sl: float = 1.0,
        dip_cooldown_h: float = 24.0,
        dip_ret_max: float = -3.0,
        dip_body_min: float = 0.2,
        dip_vol_min: float = 1.2,
        dip_rsi_low: float = 25.0,
        dip_rsi_high: float = 50.0,
        dip_regime_min: float = 0.0,
        dip_atr_max: float = 1.5,
        min_squeeze_bars: int = 7,
    ) -> None:
        self.leverage = leverage
        self.short_tp = short_tp
        self.short_sl = short_sl
        self.short_cooldown_h = short_cooldown_h
        self.short_rsi_floor = short_rsi_floor
        self.short_atr_max = short_atr_max
        self.long_tp = long_tp
        self.long_sl = long_sl
        self.long_cooldown_h = long_cooldown_h
        self.long_rsi_cap = long_rsi_cap
        self.long_regime_min = long_regime_min
        self.long_atr_max = long_atr_max
        self.dip_tp = dip_tp
        self.dip_sl = dip_sl
        self.dip_cooldown_h = dip_cooldown_h
        self.dip_ret_max = dip_ret_max
        self.dip_body_min = dip_body_min
        self.dip_vol_min = dip_vol_min
        self.dip_rsi_low = dip_rsi_low
        self.dip_rsi_high = dip_rsi_high
        self.dip_regime_min = dip_regime_min
        self.dip_atr_max = dip_atr_max
        self.min_squeeze_bars = min_squeeze_bars

        self._client: LiveMarketClient | None = None
        self._last_short: dict[str, datetime] = {}
        self._last_long: dict[str, datetime] = {}
        self._last_dip: dict[str, datetime] = {}

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        print(
            f"SqueezeV6B initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"SHORT TP/SL={self.short_tp}/{self.short_sl}% "
            f"CD={self.short_cooldown_h}h RSI>={self.short_rsi_floor} ATR<={self.short_atr_max} | "
            f"LONG TP/SL={self.long_tp}/{self.long_sl}% "
            f"CD={self.long_cooldown_h}h RSI<={self.long_rsi_cap} "
            f"regime>={self.long_regime_min}% ATR<={self.long_atr_max} | "
            f"DIP TP/SL={self.dip_tp}/{self.dip_sl}% "
            f"CD={self.dip_cooldown_h}h ret24<={self.dip_ret_max}% "
            f"body>={self.dip_body_min} vol>={self.dip_vol_min} "
            f"RSI={self.dip_rsi_low}-{self.dip_rsi_high} "
            f"regime>={self.dip_regime_min}% ATR<={self.dip_atr_max} | "
            f"min_sq={self.min_squeeze_bars}",
            file=sys.stderr,
        )

    def poll(self) -> list[Signal] | None:
        assert self._client is not None
        now = datetime.now(UTC)
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

        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval="1h",
            start=start,
            end=now,
        )
        if len(candles) < _WARMUP_BARS + 1:
            return []

        df = _candles_to_df(candles)
        df = _compute_indicators(df)

        last_idx = len(df) - 1
        if df.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx < _WARMUP_BARS:
            return []

        row = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]

        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            return []

        rsi = row.get("rsi_14", 50.0)
        if pd.isna(rsi):
            rsi = 50.0

        atr_ratio = row.get("atr_ratio", 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0

        ret_72h = row.get("ret_72h", 0.0)
        if pd.isna(ret_72h):
            ret_72h = 0.0

        ret_24h = row.get("ret_24h", 0.0)
        if pd.isna(ret_24h):
            ret_24h = 0.0

        vol_ratio = row.get("vol_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0

        body_ratio = row.get("body_ratio", 0.0)
        if pd.isna(body_ratio):
            body_ratio = 0.0

        mom = row["mom_slope"]
        squeeze_breakout = prev["squeeze_count"] >= self.min_squeeze_bars and not row["squeeze_on"]

        # Preserve the backtest ordering: SHORT -> LONG -> DIP.
        last_short = self._last_short.get(symbol)
        short_ok = (
            last_short is None or
            (now - last_short).total_seconds() >= self.short_cooldown_h * 3600
        )
        if (
            squeeze_breakout and
            atr_ratio <= self.short_atr_max and
            short_ok and
            mom < 0 and
            rsi >= self.short_rsi_floor
        ):
            self._last_short[symbol] = now
            return [Signal(
                signal_date=datetime.now(UTC),
                position_type=PositionType.SHORT,
                ticker=symbol,
                tp_pct=self.short_tp,
                sl_pct=self.short_sl,
                leverage=self.leverage,
                market_type=MarketType.FUTURES,
                taker_fee_rate=TAKER_FEE,
                metadata={
                    "strategy": "squeeze_v6b_short",
                    "mom": round(float(mom), 6),
                    "rsi": round(float(rsi), 1),
                    "atr_ratio": round(float(atr_ratio), 2),
                    "sq_count": int(prev["squeeze_count"]),
                },
            )]

        last_long = self._last_long.get(symbol)
        long_ok = (
            last_long is None or
            (now - last_long).total_seconds() >= self.long_cooldown_h * 3600
        )
        if (
            squeeze_breakout and
            atr_ratio <= self.long_atr_max and
            long_ok and
            mom > 0 and
            rsi <= self.long_rsi_cap and
            ret_72h >= self.long_regime_min
        ):
            self._last_long[symbol] = now
            return [Signal(
                signal_date=datetime.now(UTC),
                position_type=PositionType.LONG,
                ticker=symbol,
                tp_pct=self.long_tp,
                sl_pct=self.long_sl,
                leverage=self.leverage,
                market_type=MarketType.FUTURES,
                taker_fee_rate=TAKER_FEE,
                metadata={
                    "strategy": "squeeze_v6b_long",
                    "mom": round(float(mom), 6),
                    "rsi": round(float(rsi), 1),
                    "atr_ratio": round(float(atr_ratio), 2),
                    "ret_72h": round(float(ret_72h), 1),
                    "sq_count": int(prev["squeeze_count"]),
                },
            )]

        last_dip = self._last_dip.get(symbol)
        dip_ok = (
            last_dip is None or
            (now - last_dip).total_seconds() >= self.dip_cooldown_h * 3600
        )
        dip_atr_ok = self.dip_atr_max == 0 or atr_ratio <= self.dip_atr_max
        if (
            dip_ok and
            dip_atr_ok and
            ret_24h <= self.dip_ret_max and
            body_ratio >= self.dip_body_min and
            vol_ratio >= self.dip_vol_min and
            self.dip_rsi_low <= rsi <= self.dip_rsi_high and
            ret_72h >= self.dip_regime_min
        ):
            self._last_dip[symbol] = now
            return [Signal(
                signal_date=datetime.now(UTC),
                position_type=PositionType.LONG,
                ticker=symbol,
                tp_pct=self.dip_tp,
                sl_pct=self.dip_sl,
                leverage=self.leverage,
                market_type=MarketType.FUTURES,
                taker_fee_rate=TAKER_FEE,
                metadata={
                    "strategy": "squeeze_v6b_dip",
                    "rsi": round(float(rsi), 1),
                    "atr_ratio": round(float(atr_ratio), 2),
                    "ret_72h": round(float(ret_72h), 1),
                    "ret_24h": round(float(ret_24h), 1),
                    "vol_ratio": round(float(vol_ratio), 2),
                    "body_ratio": round(float(body_ratio), 3),
                },
            )]

        return []
