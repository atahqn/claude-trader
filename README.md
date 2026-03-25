# claude-trader

This codebase is designed for LLMs to draft and backtest their strategies and to later deploy them.

## Strategy Status

- Latest recommended strategy iteration: **V7**
- Current deployed/live implementation baseline: **V6** (`live/squeeze_v6_strategy.py`) with `V6B` (`live/squeeze_v6b_strategy.py`) also available
- V7 is a minimal update over V6: the SHORT logic is unchanged, and the LONG signal keeps the same squeeze + bull-regime rules but widens TP/SL from **3.0% / 1.5%** to **4.0% / 2.0%**
- Treat [`STRATEGY_EVOLUTION.md`](STRATEGY_EVOLUTION.md) as the source of truth for research history, current recommendation, and evaluation caveats

## Backtester Architecture

The shared backtester now supports a staged research pipeline instead of forcing
all work through per-signal `backtest_signal()` fetches:

1. `prepare_market_context(...)`
   Fetch hourly market data once per symbol for the backtest interval plus
   warmup, with optional listing-history futures datasets via
   `MarketDataRequest` (`funding_rates`, `mark_price_klines`,
   `premium_index_klines`).
2. Feature stage
   Strategies compute aligned symbol-level features once from prepared hourly
   context.
3. Signal stage
   Strategies generate candidate `Signal` objects from prepared context instead
   of fetching candles internally.
4. Portfolio stage
   `backtest_portfolio(...)` now schedules candidates with the candle-based
   approximate path first, then resolves only accepted trades exactly.
5. Execution stage
   Exact resolution uses a shared `BacktestExecutionSession` with chunked lazy
   caches for `1m` candles and `aggTrades`, so overlapping trades reuse the
   same exact data windows.

### New Shared Entry Points

- `backtester.prepare_market_context`
- `backtester.generate_signals_from_prepared_context`
- `backtester.BacktestExecutionSession`

`backtest_signal()` remains available as the compatibility fallback for direct
single-signal evaluation.

`backtest_portfolio(..., legacy_scheduler=True)` keeps the previous
resolve-while-scheduling behavior if an existing experiment still depends on
it.

## Live Functionality

The live layer is responsible for:

- loading exchange credentials and runtime settings
- polling signal generators
- checking capital and position-slot availability before execution
- submitting entries and protective exits
- tracking open positions and timeouts
- reconciling local state with exchange state

Configuration can be loaded from environment variables or
`~/.claude_trader/live_config.json`. The live runner scripts also accept
`--config /path/to/live_config.json` to read a specific JSON config file.

Environment variables:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_BASE_URL`
- `BINANCE_POSITION_SIZE`
- `BINANCE_MAX_POSITIONS`
- `BINANCE_MAX_HOLDING_HOURS`
- `BINANCE_ORDER_CHECK_INTERVAL`
- `BINANCE_TESTNET`
