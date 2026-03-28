# claude-trader
Bu kod, Cevat Ticari Şirketinin mülkiyetindedir. Şirketin yazılı izni (cevat.ticari@cevat.com) olmadan kullanılması, Türkiye Cumhuriyeti yasalarının 3149'uncu maddesinin 7'nci fıkrası uyarınca kesinlike yasaktır 😂.

This codebase is designed for LLMs to draft and backtest their strategies and to later deploy them.

## Strategy Status

- Recommended strategy iteration: **V8.1** (V8 signals + 72h max holding time)
- Recommended trading space:
  `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, `XRP/USDT`, `DOGE/USDT`,
  `AVAX/USDT`, `LINK/USDT`, `ENA/USDT`, `INJ/USDT`, `NEAR/USDT`,
  `ALGO/USDT`, `RENDER/USDT`, `WIF/USDT`, `ADA/USDT`, `APT/USDT`
- Canonical evaluation path:
  `backtester.StrategyEvaluator` with
  `DEVELOPMENT_WINDOWS`, `EVALUATION_WINDOWS`, or `ALL_WINDOWS`
- Treat [`STRATEGY_EVOLUTION.md`](STRATEGY_EVOLUTION.md) as the source of truth for research history and evaluation caveats

## Backtester Architecture

The shared backtester now supports a staged research pipeline instead of forcing
all work through per-signal `backtest_signal()` fetches:

1. `prepare_market_context(...)`
   Fetch hourly market data once per symbol for the backtest interval plus
   warmup, with optional listing-history futures datasets via
   `MarketDataRequest` (`funding_rates`, `mark_price_klines`,
   `premium_index_klines`). If a strategy requests a lower
   `poll_interval` than its analysis interval, the prepared context also
   carries the lower-timeframe poll candles needed for preview-style replay.
2. Feature stage
   Strategies compute aligned symbol-level features once from prepared hourly
   context.
3. Signal stage
   Strategies generate candidate `Signal` objects from prepared context instead
   of fetching candles internally. Live strategies that support historical
   replay expose `generate_backtest_signals(...)` through the shared
   `SignalGenerator` interface. This supports both close-only replay and
   lower poll-interval replay such as `1h` analysis with `15m`, `5m`, or
   `1m` polling.
4. Evaluation stage
   `StrategyEvaluator` groups windows into contiguous fetch periods, reuses a
   shared `BacktestExecutionSession`, and resolves each scored window through
   `backtest_signals(...)`.
5. Validation stage
   `validate_no_lookahead(...)` replays sampled signals against truncated
   market context to catch future-data leaks in signal generation.
6. Execution stage
   Exact resolution uses a shared `BacktestExecutionSession` with chunked lazy
   caches for `1m` candles and `aggTrades`, so overlapping trades reuse the
   same exact data windows.

### New Shared Entry Points

- `backtester.StrategyEvaluator`
- `backtester.validate_no_lookahead`
- `backtester.prepare_market_context`
- `backtester.BacktestExecutionSession`
- `backtester.DEVELOPMENT_WINDOWS`
- `backtester.EVALUATION_WINDOWS`
- `backtester.ALL_WINDOWS`

`backtest_signal()` remains available as the compatibility fallback for direct
single-signal evaluation.

### Evaluation Standard

Going forward, new strategy evaluation should use the shared evaluator and the
coded window calendar instead of one-off `*_eval.py` scripts.

- Development: `DEVELOPMENT_WINDOWS`
- Holdout + secondary OOS: `EVALUATION_WINDOWS`
- Full report: `ALL_WINDOWS`
- One-time signal-generation bias check per strategy version:
  `validate_no_lookahead(...)`

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
