# marketdata

Reference for all available market data, technical indicators, and structural price levels.

## Binance Datasets

5 dataset types via `MarketDataRequest` (`marketdata/models.py`). OHLCV is always required.

| Dataset | Enum value | Columns merged onto frame | Notes |
|---------|-----------|---------------------------|-------|
| OHLCV | `OHLCV` | `open_time`, `close_time`, `open`, `high`, `low`, `close`, `volume`, `taker_buy_volume` | Always included. Default 1h interval. |
| Funding rates | `FUNDING_RATES` | `funding_timestamp`, `funding_rate`, `funding_mark_price` | 8h frequency, merged asof. |
| Mark price klines | `MARK_PRICE_KLINES` | `mark_open`, `mark_high`, `mark_low`, `mark_close`, `mark_volume` (+ `mark_open_time`, `mark_close_time`) | Same interval as OHLCV. |
| Premium index klines | `PREMIUM_INDEX_KLINES` | `premium_open`, `premium_high`, `premium_low`, `premium_close`, `premium_volume` (+ `premium_open_time`, `premium_close_time`) | Same interval as OHLCV. |
| Agg trades | `AGG_TRADES` | Not merged onto frame. Access via `ctx.raw(DataRequirement.AGG_TRADES)`. | Only ~1 year of history. Not suitable for signal generation. |

### Requesting data

```python
from marketdata.models import MarketDataRequest, DataRequirement

# OHLCV only (default)
MarketDataRequest.ohlcv_only()

# OHLCV + funding + premium + key levels
MarketDataRequest(
    datasets=frozenset({DataRequirement.OHLCV, DataRequirement.FUNDING_RATES, DataRequirement.PREMIUM_INDEX_KLINES}),
    include_key_levels=True,
)
```

Non-OHLCV datasets are merged onto the OHLCV frame via `merge_asof` on `close_time` (backward direction). Each row in the frame carries the latest available value for each supplementary dataset.


## Technical Indicators (`backtester/indicators.py`)

Compute with `compute_indicator_frame(df, ["indicator_name", ...])`. Dependencies are auto-resolved and internal columns are dropped from output. Use `required_warmup(["indicator_name", ...])` to get the number of leading bars needed.

| Indicator | Description |
|-----------|-------------|
| `rsi_14` | 14-period RSI (0-100) |
| `atr_14` | 14-period Average True Range |
| `atr_72_avg` | 72-bar rolling mean of `atr_14` |
| `atr_ratio` | `atr_14 / atr_72_avg` (volatility expansion/contraction) |
| `ret_24h` | 24-bar percent return |
| `ret_48h` | 48-bar percent return |
| `ret_72h` | 72-bar percent return |
| `vol_sma_20` | 20-bar volume SMA |
| `vol_ratio` | `volume / vol_sma_20` |
| `ema_20` | 20-period EMA of close |
| `bb_upper` | Bollinger upper band (20-period SMA + 2 std) |
| `bb_lower` | Bollinger lower band (20-period SMA - 2 std) |
| `bb_pct_b` | %B: position within Bollinger Bands (0 = lower, 1 = upper) |
| `bb_width` | Band width normalized by middle band |
| `kc_upper` | Keltner upper (EMA-20 + 1.5 x ATR-14) |
| `kc_lower` | Keltner lower (EMA-20 - 1.5 x ATR-14) |
| `squeeze_on` | `True` when Bollinger Bands are inside Keltner Channels |
| `squeeze_count` | Consecutive bars in squeeze (resets to 0 on release) |
| `mom_slope` | 20-bar linear regression slope of close |
| `body` | Candle body: `close - open` |
| `body_ratio` | `body / (high - low)` |
| `adx_14` | 14-period ADX (trend strength, 0-100) |
| `volume_delta` | Net taker aggression: `2 * taker_buy_volume - volume` |
| `cvd` | Cumulative volume delta (running sum of `volume_delta`) |
| `t3` | Tilson T3 moving average (period 5, volume factor 0.7). Six cascaded EMAs combined with weighted coefficients for ultra-smooth trend following with minimal lag. |

### Usage

```python
from backtester.indicators import compute_indicator_frame, required_warmup

warmup = required_warmup(["rsi_14", "squeeze_on", "t3"])
frame = compute_indicator_frame(ohlcv_frame, ["rsi_14", "squeeze_on", "t3"])
# frame now has rsi_14, squeeze_on, t3 columns added
```


## Key Levels (`marketdata/key_levels.py`)

Structural price levels across multiple timeframes, computed from 4H/D/W/M candles. All levels are lookahead-free: derived only from completed periods whose `close_time <= query timestamp`.

Enable via `MarketDataRequest(..., include_key_levels=True)`. The backtester pipeline fetches the required candle data automatically.

### Available levels

| Group | Fields | Description |
|-------|--------|-------------|
| 4-Hour | `h4_open`, `prev_h4_high`, `prev_h4_low`, `h4_eq` | Current 4H open + previous 4H range and midpoint |
| Daily | `daily_open`, `pdh`, `pdl`, `daily_eq` | Current day open + previous day high/low/midpoint |
| Weekly | `weekly_open`, `prev_week_high`, `prev_week_low`, `weekly_eq` | Current week open + previous week range |
| Monthly | `monthly_open`, `prev_month_high`, `prev_month_low`, `monthly_eq` | Current month open + previous month range |
| Quarterly | `quarterly_open`, `prev_quarter_high`, `prev_quarter_low`, `quarterly_eq` | Aggregated from daily candles |
| Yearly | `yearly_open`, `yearly_high`, `yearly_low`, `yearly_eq` | Running cumulative year high/low |
| Monday | `monday_high`, `monday_low`, `monday_mid` | Completed Monday range (available from Tuesday onward) |
| Asia session | `asia_open`, `asia_high`, `asia_low` | Most recent completed Asia session (00:00-08:00 UTC) |
| London session | `london_open`, `london_high`, `london_low` | Most recent completed London session (08:00-16:00 UTC) |
| NY session | `ny_open`, `ny_high`, `ny_low` | Most recent completed NY session (13:00-22:00 UTC) |

All values are `None` when insufficient history exists (e.g. no completed previous period yet).

### Usage (backtesting)

Key levels are stored on `PreparedMarketContext` and accessed via `get_key_levels(symbol, timestamp)`:

```python
class MyStrategy(SignalGenerator):
    def market_data_request(self):
        return MarketDataRequest.ohlcv_only(include_key_levels=True)

    def generate_backtest_signals(self, ctx, symbols, start, end):
        for symbol in symbols:
            candles = ctx.slice_analysis_candles(symbol, start, end)
            for c in candles:
                kl = ctx.get_key_levels(symbol, c.close_time)
                if kl and kl.pdh and c.close > kl.pdh:
                    # price broke above previous day high
                    ...
```
