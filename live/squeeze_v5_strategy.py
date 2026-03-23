"""Live signal generator: Squeeze-Only V5 Strategy.

Ports the V5 strategy (developed March 2026) to the live framework.

Architecture:
  Single sub-strategy — Squeeze Breakout SHORT only.
  No vol spike. No regime gate (squeeze works in all non-extreme-bull regimes).

  SQUEEZE BREAKOUT (SHORT ONLY):
    - Bollinger Bands compress inside Keltner Channel for 7+ consecutive hours
    - On squeeze release (BB expands outside KC), enter SHORT if:
      - 20-bar momentum slope is negative (downward breakout direction)
      - RSI_14 >= 30 (not already oversold — avoids mean-reversion traps)
      - ATR_14 / 72-bar avg ATR <= 1.3 (skip elevated volatility — false breakouts)
    - TP: 2.0%, SL: 1.0% (2:1 risk-reward)
    - Per-symbol cooldown: 12 hours

  KEY CHANGES vs V3:
    1. Vol spike REMOVED (liability: -92% across 7 walk-forward windows)
    2. Min squeeze bars raised from 5 to 7 (higher quality compression)
    3. Regime gate REMOVED (squeeze works in all conditions; signal logic
       already filters via momentum direction + RSI)
    4. ATR ratio gate ADDED (skip when ATR ratio > 1.3 — chaotic markets
       produce false squeeze breakouts)

  VALIDATION (31 1-week windows, Apr 2024 — Mar 2026):
    - Weekly win rate: 71% (22/31 weeks profitable)
    - Total PNL: +69.04% across all windows
    - Average weekly PNL: +2.23%
    - Worst single week: -5.00%
    - Max drawdown in any week: 5.85%

  vs V3 on same 31 windows:
    - Weekly win rate: 71% vs 52% (+20pp improvement)
    - Worst week: -5% vs -40% (8x less severe)
    - Max drawdown: 5.85% vs 33.24% (5.7x lower)

Look-ahead bias prevention:
  - All signals fire at candle close_time (end of candle period)
  - All indicators are backward-looking: rolling(), ewm(), shift(1), pct_change()
  - No center=True, no forward-looking operations
  - RSI uses Wilder smoothing (ewm alpha=1/14) — backward-looking
  - Entry executes via market order after signal (live) or close_time+5s (backtest)
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
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
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


class SqueezeV5Strategy(SignalGenerator):
    """Squeeze-only V5 live signal generator.

    Trades only squeeze breakout SHORT signals. No vol spike.
    Validated across 31 1-week windows (Apr 2024 — Mar 2026): 71% weekly WR.
    """

    def __init__(
        self,
        leverage: float = 1.0,
        tp_pct: float = 2.0,
        sl_pct: float = 1.0,
        cooldown_h: float = 12.0,
        rsi_floor: float = 30.0,
        min_squeeze_bars: int = 7,
        atr_ratio_max: float = 1.3,
    ) -> None:
        self.leverage = leverage
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.cooldown_h = cooldown_h
        self.rsi_floor = rsi_floor
        self.min_squeeze_bars = min_squeeze_bars
        self.atr_ratio_max = atr_ratio_max

        self._client: LiveMarketClient | None = None
        self._last_signal: dict[str, datetime] = {}

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        print(
            f"SqueezeV5 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"SHORT-only TP/SL={self.tp_pct}/{self.sl_pct}% "
            f"CD={self.cooldown_h}h RSI>={self.rsi_floor} "
            f"min_sq={self.min_squeeze_bars} ATR_ratio<={self.atr_ratio_max}",
            file=sys.stderr,
        )

    def poll(self) -> list[Signal] | None:
        assert self._client is not None
        now = datetime.now(UTC)
        signals: list[Signal] = []

        for symbol in SYMBOLS:
            try:
                sig = self._check_symbol(symbol, now)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                print(f"Error checking {symbol}: {exc}", file=sys.stderr)

        return signals if signals else None

    def _check_symbol(self, symbol: str, now: datetime) -> Signal | None:
        assert self._client is not None

        # Per-symbol cooldown
        last = self._last_signal.get(symbol)
        if last is not None and (now - last).total_seconds() < self.cooldown_h * 3600:
            return None

        # Fetch candles
        start = now - timedelta(hours=_LOOKBACK_BARS + 2)
        candles = self._client.fetch_klines(
            symbol=symbol.replace("/", ""),
            interval="1h",
            start=start,
            end=now,
        )

        if len(candles) < _WARMUP_BARS + 1:
            return None

        df = _candles_to_df(candles)
        df = _compute_indicators(df)

        # Evaluate last completed candle
        last_idx = len(df) - 1
        if df.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx < _WARMUP_BARS:
            return None

        row = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]

        # Check required indicators
        if pd.isna(row.get("mom_slope")) or pd.isna(row.get("atr_ratio")):
            return None

        rsi = row.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50

        atr_ratio = row.get("atr_ratio", 1.0)
        if pd.isna(atr_ratio):
            atr_ratio = 1.0

        # ATR ratio gate: skip in elevated volatility
        if atr_ratio > self.atr_ratio_max:
            return None

        # Squeeze breakout: previous bar in squeeze for 7+ bars, current bar NOT in squeeze
        if prev["squeeze_count"] < self.min_squeeze_bars or row["squeeze_on"]:
            return None

        mom = row["mom_slope"]

        # SHORT only: momentum must be negative AND RSI not oversold
        if mom >= 0 or rsi < self.rsi_floor:
            return None

        # Signal!
        self._last_signal[symbol] = now
        return Signal(
            signal_date=datetime.now(UTC),
            position_type=PositionType.SHORT,
            ticker=symbol,
            tp_pct=self.tp_pct,
            sl_pct=self.sl_pct,
            leverage=self.leverage,
            market_type=MarketType.FUTURES,
            taker_fee_rate=0.0005,
            metadata={
                "strategy": "squeeze_v5",
                "mom": round(float(mom), 6),
                "rsi": round(float(rsi), 1),
                "atr_ratio": round(float(atr_ratio), 2),
                "sq_count": int(prev["squeeze_count"]),
            },
        )
