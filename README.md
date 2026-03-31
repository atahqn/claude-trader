# claude-trader
Bu kod, Cevat Ticari Şirketinin mülkiyetindedir. Şirketin yazılı izni (cevat.ticari@cevat.com) olmadan kullanılması, Türkiye Cumhuriyeti yasalarının 3149'uncu maddesinin 7'nci fıkrası uyarınca kesinlike yasaktır 😂.

This codebase is designed for LLMs to draft and backtest their strategies and to later deploy them.


## Backtester Architecture

The shared backtester now supports a staged research pipeline:

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

Strategy evaluation should use the shared evaluator and the
coded window calendar instead of one-off `*_eval.py` scripts.

- Development: `DEVELOPMENT_WINDOWS`
- Evaluation: `EVALUATION_WINDOWS`
- Full report: `ALL_WINDOWS`
- One-time signal-generation bias check per strategy version:
  `validate_no_lookahead(...)`

### Drawdown Semantics

When `run_strategy_eval.py` prints the category table, each row's `DD` is
recomputed from the resolved trades in that category only, in chronological
order after filtering to that category. The summary line's `MDD` is the `ALL`
row drawdown, computed from the full chronological trade stream across every
selected window.


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
JSON configs can reference `live/local_keys.py` via `api_key_var` /
`api_secret_var` so secrets stay out of tracked files. Signal-generation
parameters such as ATR gates, TP/SL percentages, RSI thresholds, and other
strategy logic stay in strategy code and generated signals, while live runtime
controls stay in live config.

Environment variables:

- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_BASE_URL`
- `BYBIT_POSITION_SIZE`
- `BYBIT_MAX_POSITIONS`
- `BYBIT_ORDER_CHECK_INTERVAL`
- `BYBIT_TESTNET`
