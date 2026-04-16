# btc_structure

Daily BTC market structure engine. Detects swing highs/lows, ranks them by scope, computes Fibonacci retracement/extension levels, and produces structural regime features from daily OHLCV data.

## Data exposed

This module exposes **daily BTC structure features** via `DailyStructureProvider`. These are derived exclusively from daily OHLCV candles (the same dataset type listed in `program.md` under "What to Do"). No additional Binance datasets (agg_trades, funding_rates, mark_price_klines, premium_index_klines) are required.

Strategies declare which columns they need. Only the computation layers required for those columns are evaluated.

### STRUCTURE_REGIME (7 columns)

Macro direction filters. Use to gate entries on/off.

| Column | Type | Description |
|--------|------|-------------|
| `market_bias_after_close` | str | Swing-based bias: `"bullish"` / `"bearish"` / `"neutral"` |
| `major_global_bullish_confluence_flag` | bool | Major + global scopes both agree bullish (break + fib leg) |
| `major_global_bearish_confluence_flag` | bool | Major + global scopes both agree bearish |
| `global_continuation_long_flag` | bool | Global break bullish + fib leg bullish + price above midpoint |
| `global_continuation_short_flag` | bool | Inverse |
| `major_last_break_is_bullish` | bool | Last major-scope structure break was upward |
| `global_last_break_is_bullish` | bool | Last global-scope structure break was upward |

### STRUCTURE_LEVELS (10 columns)

Structural price levels. Use for entry zones, stops, and targets.

| Column | Type | Description |
|--------|------|-------------|
| `major_fib_leg_direction` | str | `"bullish"` or `"bearish"` |
| `major_fib_leg_position` | float | 0-1: where price sits in the major fib range |
| `major_fib_0_5_level` | float | Major fib 50% price level |
| `major_fib_0_618_level` | float | Major fib 61.8% (golden ratio) price level |
| `major_fib_0_66_level` | float | Major fib 66% price level |
| `major_fib_0_34_level` | float | Major fib 34% price level (stop zone for longs) |
| `major_fib_ext_up_1_618_level` | float | 1.618 extension above (take-profit zone) |
| `major_pullback_long_candidate_flag` | bool | Price in golden pocket + bullish confluence |
| `major_pullback_short_candidate_flag` | bool | Inverse |
| `global_fib_leg_position` | float | Same as major but at multi-year scope |

### STRUCTURE_EVENTS (6 columns)

Bar-level event flags. Fire on the specific daily close where the event occurred.

| Column | Type | Description |
|--------|------|-------------|
| `choch_up_on_close_flag` | bool | Change of character upward (reversal) |
| `choch_down_on_close_flag` | bool | Change of character downward |
| `bos_up_on_close_flag` | bool | Break of structure upward (continuation) |
| `bos_down_on_close_flag` | bool | Break of structure downward |
| `confirmed_high_on_close_flag` | bool | Swing high confirmed on this bar |
| `confirmed_low_on_close_flag` | bool | Swing low confirmed on this bar |

### Full matrix

When `columns=None`, the provider computes ~510 columns across 4 scopes (local, structural, major, global) including per-scope fib levels, distances, zone flags, rolling break pressure, and cross-scope alignment features.

## Usage

### Backtesting

```python
from btc_structure import DailyStructureProvider, STRUCTURE_REGIME, STRUCTURE_LEVELS

class MyStrategy(SignalGenerator):
    _STRUCTURE_COLUMNS = STRUCTURE_REGIME + STRUCTURE_LEVELS

    def __init__(self):
        self._structure = DailyStructureProvider(columns=self._STRUCTURE_COLUMNS)

    def generate_backtest_signals(self, ctx, symbols, start, end):
        data_cutoff = ctx.for_symbol("BTC/USDT").frame["close_time"].max()
        self._structure.ensure_computed_until(data_cutoff)

        frame = ctx.for_symbol("BTC/USDT").frame
        frame = compute_indicator_frame(frame, ["rsi_14", "atr_14"])
        frame = self._structure.merge_onto(
            frame, self._STRUCTURE_COLUMNS, cutoff=data_cutoff,
        )
        # frame now has both hourly indicators and daily structure features
```

### Live

```python
def setup(self, client):
    self._structure = DailyStructureProvider(client, columns=STRUCTURE_REGIME)
    self._structure.refresh_if_stale(self.current_time())

def poll(self):
    self._structure.refresh_if_stale(self.current_time())
    daily = self._structure.latest()
    if not daily["global_continuation_long_flag"]:
        return None
```


## Engine API

`simulate_btc_structure(ohlcv, config, *, checkpoint=None)` returns a
`tuple[StructureArtifacts, StructureCheckpoint]`. The checkpoint captures the
engine's internal state so that subsequent calls with appended bars can resume
from the last processed bar instead of re-running the full history. Pass
`checkpoint=None` (the default) for a fresh computation.

```python
from btc_structure import BtcStructureConfig, simulate_btc_structure

config = BtcStructureConfig.for_interval("1d")
artifacts, checkpoint = simulate_btc_structure(ohlcv_day_1_to_100, config)

# Later, with more bars appended:
artifacts, checkpoint = simulate_btc_structure(
    ohlcv_day_1_to_200, config, checkpoint=checkpoint,
)
```

The checkpoint is validated on resume via a SHA-256 prefix hash over the OHLCV
data and a config fingerprint. If either mismatches (e.g. different data or
config), the checkpoint is discarded and full recomputation runs automatically.

`DailyStructureProvider` uses this internally — no changes needed in strategy
code.

## Requirements

- Daily OHLCV data from BTC/USDT perpetual (Binance futures), listing date (2019-09-08) to present
- Uses the existing `BinanceClient` from `backtester/data.py` -- no additional data provider needed
- Minimum 3 years of history for meaningful output, 5 years ideal
