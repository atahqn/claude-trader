"""Live signal generator: RSI-Filtered Asymmetric Regime Ensemble (Strategy V3).

Ports strategy_v2/FINAL_STRATEGY_V3.py to the live framework.

Two sub-strategies with a 72h-return asymmetric regime detector and RSI quality filters:
  - Squeeze Breakout (SHORT ONLY): BB inside KC for 5+ bars -> short on expansion
    Active when ret_72h < 2% (bearish + neutral). RSI >= 30 required. TP=2.0%, SL=1.0%, CD=12h.
  - Volume Spike Momentum: >1.8x vol + >50% body -> trade direction
    Active ONLY when ret_72h > 0% (bullish). Skip LONG if RSI > 75. TP=2.0%, SL=1.5%, CD=4h.

Key improvements over V2 (asymmetric_regime_strategy.py):
  1. RSI filter on squeeze: skip SHORT when RSI<30 (avoids oversold mean-reversion traps)
  2. RSI filter on vol spike: skip LONG when RSI>75 (avoids overbought exhaustion)
  3. Lower vol spike threshold (1.8x vs 2.0x) - more signals, RSI maintains quality

Polls every hour (aligned to candle close). On each poll, fetches the latest
100 hourly candles, computes indicators, and checks if the most recent
completed candle triggers a signal.
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

# 72h regime + 20-bar rolling + buffer
_LOOKBACK_BARS = 100
_WARMUP_BARS = 80


def _candles_to_df(candles: list) -> pd.DataFrame:
    """Convert backtester Candle objects to a DataFrame."""
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
    """Compute all indicators needed by the V3 strategy.

    All operations are backward-looking only (no look-ahead).
    """
    df = df.copy()
    c = df["close"]

    # RSI 14 (Wilder smoothing) — used for quality filtering
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss))

    # Volume ratio (20-bar SMA)
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]

    # ATR 14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # 72h return for regime detection (in percent)
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

    # Momentum slope (linear regression over 20 bars)
    df["mom_slope"] = c.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # Body ratio for vol spike detection
    rng = df["high"] - df["low"]
    df["body_ratio"] = (c - df["open"]) / rng.replace(0, np.nan)

    return df


class RsiFilteredStrategy(SignalGenerator):
    """Live signal generator for the V3 RSI-filtered asymmetric regime ensemble."""

    def __init__(
        self,
        leverage: float = 1.0,
        squeeze_tp: float = 2.0,
        squeeze_sl: float = 1.0,
        squeeze_cooldown_h: float = 12.0,
        squeeze_rsi_floor: float = 30.0,
        vs_tp: float = 2.0,
        vs_sl: float = 1.5,
        vs_cooldown_h: float = 4.0,
        vs_vol_min: float = 1.8,
        vs_body_min: float = 0.5,
        vs_rsi_long_cap: float = 75.0,
        min_squeeze_bars: int = 5,
        squeeze_regime_thresh: float = 2.0,
        vs_regime_thresh: float = 0.0,
    ) -> None:
        self.leverage = leverage
        self.squeeze_tp = squeeze_tp
        self.squeeze_sl = squeeze_sl
        self.squeeze_cooldown_h = squeeze_cooldown_h
        self.squeeze_rsi_floor = squeeze_rsi_floor
        self.vs_tp = vs_tp
        self.vs_sl = vs_sl
        self.vs_cooldown_h = vs_cooldown_h
        self.vs_vol_min = vs_vol_min
        self.vs_body_min = vs_body_min
        self.vs_rsi_long_cap = vs_rsi_long_cap
        self.min_squeeze_bars = min_squeeze_bars
        self.squeeze_regime_thresh = squeeze_regime_thresh
        self.vs_regime_thresh = vs_regime_thresh

        self._client: LiveMarketClient | None = None
        # Per-symbol cooldown tracking: symbol -> last signal datetime
        self._last_squeeze: dict[str, datetime] = {}
        self._last_vs: dict[str, datetime] = {}

    def setup(self, client: LiveMarketClient) -> None:
        self._client = client
        print(
            f"RsiFilteredStrategy V3 initialized | "
            f"symbols={len(SYMBOLS)} | leverage={self.leverage}x | "
            f"squeeze SHORT-only TP/SL={self.squeeze_tp}/{self.squeeze_sl}% "
            f"CD={self.squeeze_cooldown_h}h RSI>={self.squeeze_rsi_floor} | "
            f"vol_spike TP/SL={self.vs_tp}/{self.vs_sl}% "
            f"CD={self.vs_cooldown_h}h vol>={self.vs_vol_min}x RSI_long<={self.vs_rsi_long_cap} | "
            f"regime: squeeze<{self.squeeze_regime_thresh}% vs>{self.vs_regime_thresh}%",
            file=sys.stderr,
        )

    def poll(self) -> list[Signal] | None:
        """Fetch latest candles for all symbols and check for signals."""
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
        """Fetch candles and evaluate the latest completed candle for signals."""
        assert self._client is not None

        # Fetch last _LOOKBACK_BARS hourly candles
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

        # Only evaluate the last completed candle (second-to-last if current is open)
        last_idx = len(df) - 1
        if df.iloc[last_idx]["close_time"] > now:
            last_idx -= 1
        if last_idx < _WARMUP_BARS:
            return []

        row = df.iloc[last_idx]
        prev = df.iloc[last_idx - 1]

        regime = row.get("ret_72h", None)
        if pd.isna(regime) or pd.isna(row.get("vol_ratio")) or pd.isna(row.get("mom_slope")):
            return []

        rsi = row.get("rsi_14", 50)
        if pd.isna(rsi):
            rsi = 50

        signals: list[Signal] = []

        # === SQUEEZE SHORT SIGNAL ===
        # Active when regime < squeeze_regime_thresh (bearish + neutral)
        if regime < self.squeeze_regime_thresh:
            last_sq = self._last_squeeze.get(symbol)
            can_fire = (
                last_sq is None
                or (now - last_sq).total_seconds() >= self.squeeze_cooldown_h * 3600
            )

            if can_fire and prev["squeeze_count"] >= self.min_squeeze_bars and not row["squeeze_on"]:
                mom = row["mom_slope"]
                # SHORT ONLY when momentum is negative AND RSI not oversold
                if mom < 0 and rsi >= self.squeeze_rsi_floor:
                    signals.append(self._make_signal(
                        symbol, PositionType.SHORT,
                        self.squeeze_tp, self.squeeze_sl,
                        {
                            "strategy": "squeeze",
                            "regime": round(float(regime), 2),
                            "mom": round(float(mom), 6),
                            "rsi": round(float(rsi), 1),
                        },
                    ))
                    self._last_squeeze[symbol] = now

        # === VOL SPIKE SIGNAL ===
        # Active ONLY when regime > vs_regime_thresh (bullish)
        if regime > self.vs_regime_thresh:
            last_vs = self._last_vs.get(symbol)
            can_fire = (
                last_vs is None
                or (now - last_vs).total_seconds() >= self.vs_cooldown_h * 3600
            )

            if can_fire and row["vol_ratio"] >= self.vs_vol_min:
                br = row.get("body_ratio")
                if not pd.isna(br) and abs(br) >= self.vs_body_min:
                    pos = PositionType.LONG if br > 0 else PositionType.SHORT
                    # RSI filter: skip LONG if overbought (exhaustion risk)
                    if not (pos == PositionType.LONG and rsi > self.vs_rsi_long_cap):
                        signals.append(self._make_signal(
                            symbol, pos,
                            self.vs_tp, self.vs_sl,
                            {
                                "strategy": "vol_spike",
                                "regime": round(float(regime), 2),
                                "vol_ratio": round(float(row["vol_ratio"]), 2),
                                "body_ratio": round(float(br), 3),
                                "rsi": round(float(rsi), 1),
                            },
                        ))
                        self._last_vs[symbol] = now

        return signals

    def _make_signal(
        self,
        symbol: str,
        position_type: PositionType,
        tp_pct: float,
        sl_pct: float,
        metadata: dict,
    ) -> Signal:
        return Signal(
            signal_date=datetime.now(UTC),
            position_type=position_type,
            ticker=symbol,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            leverage=self.leverage,
            market_type=MarketType.FUTURES,
            taker_fee_rate=0.0005,
            metadata=metadata,
        )
