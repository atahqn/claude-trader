# claude-trader

Trading engine and backtester for signals. Repository is created so that an LLM (claude) can experiment with different crypto trading strategies and execute them with real money in binance.

The live layer loads credentials from environment variables or
`~/.claude_trader/live_config.json`. The live runner scripts also accept
`--config /path/to/live_config.json` to read a specific JSON config file.

Environment variables:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_BASE_URL`
- `BINANCE_POSITION_SIZE`
- `BINANCE_MAX_POSITIONS`
- `BINANCE_POLL_INTERVAL`
- `BINANCE_ORDER_CHECK_INTERVAL`
- `BINANCE_TESTNET`
