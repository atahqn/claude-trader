# claude-trader

Minimal backbone repository extracted from the original `claude_trader` workspace.

Included packages:

- `backtester/`: market-data client, trade resolution, and backtest engine.
- `live/`: live trading runtime primitives, authenticated Binance Futures client,
  signal-generator interface, order execution, and position tracking.

Excluded from this repository:

- research and experiment directories
- concrete live strategies and runner scripts
- cached data, result artifacts, and local state

## Python

The backbone code is stdlib-only, but it relies on Python 3.11+ features such as
`enum.StrEnum` and `datetime.UTC`.

## Live Config

The live layer loads credentials from environment variables or
`~/.claude_trader/live_config.json`.

Environment variables:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_BASE_URL`
- `BINANCE_POSITION_SIZE`
- `BINANCE_MAX_POSITIONS`
- `BINANCE_POLL_INTERVAL`
- `BINANCE_ORDER_CHECK_INTERVAL`
- `BINANCE_TESTNET`
