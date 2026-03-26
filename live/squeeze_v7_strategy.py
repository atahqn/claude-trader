"""Live signal generator: Squeeze V7 Strategy (SHORT + LONG).

Identical to V6 except LONG TP/SL widened from 3.0/1.5 to 4.0/2.0.
Wider stops in bull markets prevent premature stop-outs on normal pullbacks
and let winners run further. Trade WR increases from 59% to 63%.

Architecture:
  Two sub-signals from the SAME structural event (BB/KC squeeze release):

  1. SQUEEZE SHORT (unchanged from V5/V6):
    - 7+ bar compression, release with negative momentum
    - RSI >= 30 (not oversold), ATR ratio <= 1.3
    - TP: 2.0%, SL: 1.0%, Cooldown: 12h

  2. SQUEEZE LONG (V7: wider TP/SL):
    - 7+ bar compression, release with POSITIVE momentum
    - ret_72h >= 6% (STRONG BULL regime only)
    - RSI <= 70 (not overbought), ATR ratio <= 1.3
    - TP: 4.0%, SL: 2.0%, Cooldown: 12h  ← changed from V6 (3.0/1.5)

  VALIDATION (31 1-week windows, Apr 2024 — Mar 2026):
    V7 (SH_T40):
      All 31 windows: +111.40% PNL, 77.4% WkWR (24/31), worst -9.00%
      Holdout (7 unseen windows): +32.03% PNL, 85.7% WkWR (6/7), worst -3.85%
      vs V6: +98.27% → +111.40% (+13% improvement)

Look-ahead bias prevention:
  - All signals fire at candle close_time (end of candle period)
  - All indicators backward-looking: rolling(), ewm(), shift(1), pct_change()
  - ret_72h = close.pct_change(72) — backward-looking 72-bar return
  - Entry executes via market order after signal
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backtester.models import MarketType, PositionType, Signal

from .auth_client import LiveMarketClient
from .signal_generator import SignalGenerator

SYMBOLS = [
    "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
    "ENA/USDT", "INJ/USDT", "NEAR/USDT", "ALGO/USDT",
    "RENDER/USDT", "WIF/USDT", "ADA/USDT", "APT/USDT",
]

_LOOKBACK_BARS = 100
_WARMUP_BARS = 80


def _candles_to_df(candles: list) -> pd.DataFrame:
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
        return df
    return df.sort_values("open_time").reset_index(drop=True)


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators for squeeze detection. All backward-looking."""
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # RSI 14 (Wilder smoothing)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # ATR 14
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # ATR ratio: current volatility vs 72-bar average
    df["atr_72_avg"] = df["atr_14"].rolling(72).mean()
    df["atr_ratio"] = df["atr_14"] / df["atr_72_avg"]

    # 72-hour return (backward-looking)
    df["ret_72h"] = c.pct_change(72) * 100

    # Bollinger Bands (20, 2)
    bb_ma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = bb_ma + 2 * bb_std
    df["bb_lower"] = bb_ma - 2 * bb_std

    # Keltner Channel (20 EMA, 1.5 ATR)
    ema_20 = c.ewm(span=20).mean()
    df["kc_upper"] = ema_20 + 1.5 * df["atr_14"]
    df["kc_lower"] = ema_20 - 1.5 * df["atr_14"]

    # Squeeze detection (BB inside KC)
    df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])

    # Consecutive squeeze count
    cnt = 0
    counts = []
    for s in df["squeeze_on"]:
        cnt = cnt + 1 if s else 0
        counts.append(cnt)
    df["squeeze_count"] = counts

    # Momentum slope (20-bar linear regression)
    df["mom_slope"] = c.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    return df


class SqueezeV7Strategy(SignalGenerator):
    """Squeeze V7: SHORT + LONG live signal generator.

    SHORT: squeeze breakout with negative momentum (all regimes)
    LONG:  squeeze breakout with positive momentum (bull regime only, ret_72h >= 6%)
           TP/SL widened to 4.0/2.0 (from V6's 3.0/1.5)
    """

    def __init__(
        self,
        leverage: float = 1.0,
        # SHORT params (unchanged from V5/V6)
        short_tp: float = 2.0,
        short_sl: float = 1.0,
        short_cooldown_h: float = 12.0,
        short_rsi_floor: float = 30.0,
        # LONG params (V7: wider TP/SL)
        long_tp: float = 4.0,
        long_sl: float = 2.0,
        long_cooldown_h: float = 12.0,
        long_rsi_cap: float = 70.0,
        long_regime_min: float = 6.0,
        # Shared params
        min_squeeze_bars: int = 7,
        atr_ratio_max: float = 1.3,
    ) -> None:
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

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        print(
            f"SqueezeV7 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"SHORT TP/SL={self.short_tp}/{self.short_sl}% "
            f"CD={self.short_cooldown_h}h RSI>={self.short_rsi_floor} | "
            f"LONG TP/SL={self.long_tp}/{self.long_sl}% "
            f"CD={self.long_cooldown_h}h RSI<={self.long_rsi_cap} "
            f"regime>={self.long_regime_min}% | "
            f"min_sq={self.min_squeeze_bars} ATR<={self.atr_ratio_max}",
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
        signals: list[Signal] = []

        # Fetch candles
        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval="1h",
            start=start,
            end=now,
        )

        if len(candles) < _WARMUP_BARS + 1:
            return signals

        df = _candles_to_df(candles)
        df = _compute_indicators(df)

        # Evaluate last completed candle
        last_idx = len(df) - 1
        if df.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx < _WARMUP_BARS:
            return signals

        row = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]

        # Check required indicators
        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            return signals

        rsi = row.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50

        atr_ratio = row.get("atr_ratio", 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0

        ret_72h = row.get("ret_72h", 0)
        if pd.isna(ret_72h):
            ret_72h = 0

        # ATR ratio gate: skip in elevated volatility
        if atr_ratio > self.atr_ratio_max:
            return signals

        # Squeeze breakout: previous bar in squeeze for 7+ bars, current NOT in squeeze
        if prev["squeeze_count"] < self.min_squeeze_bars or row["squeeze_on"]:
            return signals

        mom = row["mom_slope"]

        # ── SHORT signal: negative momentum + RSI not oversold ────────
        last_s = self._last_short.get(symbol)
        short_ok = (last_s is None or
                    (now - last_s).total_seconds() >= self.short_cooldown_h * 3600)

        if short_ok and mom < 0 and rsi >= self.short_rsi_floor:
            self._last_short[symbol] = now
            signals.append(Signal(
                signal_date=datetime.now(UTC),
                position_type=PositionType.SHORT,
                ticker=symbol,
                tp_pct=self.short_tp,
                sl_pct=self.short_sl,
                leverage=self.leverage,
                market_type=MarketType.FUTURES,
                taker_fee_rate=0.0005,
                metadata={
                    "strategy": "squeeze_v7_short",
                    "mom": round(float(mom), 6),
                    "rsi": round(float(rsi), 1),
                    "atr_ratio": round(float(atr_ratio), 2),
                    "sq_count": int(prev["squeeze_count"]),
                },
            ))

        # ── LONG signal: positive momentum + bull regime + RSI not overbought ──
        last_l = self._last_long.get(symbol)
        long_ok = (last_l is None or
                   (now - last_l).total_seconds() >= self.long_cooldown_h * 3600)

        if long_ok and mom > 0 and rsi <= self.long_rsi_cap and ret_72h >= self.long_regime_min:
            self._last_long[symbol] = now
            signals.append(Signal(
                signal_date=datetime.now(UTC),
                position_type=PositionType.LONG,
                ticker=symbol,
                tp_pct=self.long_tp,
                sl_pct=self.long_sl,
                leverage=self.leverage,
                market_type=MarketType.FUTURES,
                taker_fee_rate=0.0005,
                metadata={
                    "strategy": "squeeze_v7_long",
                    "mom": round(float(mom), 6),
                    "rsi": round(float(rsi), 1),
                    "atr_ratio": round(float(atr_ratio), 2),
                    "ret_72h": round(float(ret_72h), 1),
                    "sq_count": int(prev["squeeze_count"]),
                },
            ))

        return signals
