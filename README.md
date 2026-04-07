# claude-trader

Bu kod, Cevat Ticari Şirketinin mülkiyetindedir. Şirketin yazılı izni (cevat.ticari@cevat.com) olmadan kullanılması, Türkiye Cumhuriyeti yasalarının 3149'uncu maddesinin 7'nci fıkrası uyarınca kesinlike yasaktır 😂.

A framework for drafting, backtesting, and live-deploying crypto trading
strategies on Binance Futures. The two core paths — backtesting and live
trading — share a single abstraction: the **SignalGenerator**. You write one
strategy class and it works in both contexts.

```
claude-trader/
├── backtester/                 # Backtest engine & evaluation pipeline
│   ├── models.py               # Signal, TradeResult, BacktestResult
│   ├── pipeline.py             # PreparedMarketContext, BacktestExecutionSession
│   ├── evaluator.py            # StrategyEvaluator, EvaluationReport
│   ├── engine.py               # backtest_signal(), backtest_signals()
│   ├── validation.py           # validate_no_lookahead()
│   └── data.py                 # BinanceClient (cached API fetcher)
├── live/                       # Live trading engine
│   ├── signal_generator.py     # SignalGenerator base class
│   ├── engine.py               # LiveEngine (main loop)
│   ├── executor.py             # OrderExecutor (signal → exchange orders)
│   ├── tracker.py              # PositionTracker (position lifecycle)
│   ├── models.py               # LiveConfig, LivePosition, GeneratorBudget
│   └── run.py                  # Entry point for live trading
├── marketdata/
│   └── models.py               # MarketDataRequest, SymbolMarketContext
└── run_strategy_eval.py        # Entry point for backtest evaluation
```


## SignalGenerator — The Core Abstraction

**File:** `live/signal_generator.py`

Every strategy extends `SignalGenerator`. It declares two methods that make the
strategy work across both paths:

| Method | Used by | Purpose |
|--------|---------|---------|
| `poll()` | Live engine | Returns `Signal`(s) for the current market state during live trading |
| `generate_backtest_signals(prepared_context, symbols, start, end)` | Backtester | Returns all `Signal`s for a historical window using prepared market data |

Both methods produce the same `Signal` object (`backtester/models.py`), which
carries everything needed to execute a trade: ticker, direction, TP/SL levels,
leverage, entry type, and timing constraints.

A strategy also declares:

- **`symbols`** — which tickers it trades
- **`market_data_request()`** — what raw datasets it needs (OHLCV, funding rates, mark price klines, etc.)
- **`indicator_request()`** — which indicator columns to precompute on each symbol's OHLCV frame (e.g. `("rsi_14", "bb_upper", "atr")`). The framework runs `compute_indicator_frame()` once per symbol during data preparation and caches the result in `PreparedMarketContext`; access it via `ctx.indicator_frame(symbol)`. Warmup bars for the requested indicators are added automatically.
- **`required_warmup_bars`** — how many extra bars to fetch before the signal window for indicator history (default 100). When `indicator_request()` is declared, the framework ensures at least `required_warmup(indicator_request())` bars; set this higher only if you need additional bars.
- **`analysis_interval`** / **`poll_interval`** — candle timeframe for analysis and optional faster polling
- **`cooldown_hours`** — minimum time between signals on the same symbol

### Signal

**File:** `backtester/models.py`

A `Signal` is the contract between strategy logic and execution. Key fields:

- `signal_date`, `ticker`, `position_type` (LONG/SHORT)
- `tp_pct` / `sl_pct` or `tp_price` / `sl_price` — exit levels
- `leverage`, `entry_price` (None = market order, else limit)
- `entry_delay_seconds`, `fill_timeout_seconds`, `max_holding_hours`
- `size_multiplier`, `metadata`

The same Signal object is interpreted by both the backtester's resolution
engine and the live order executor.


## Backtesting

The backtester evaluates a `SignalGenerator` across canonical time windows.
The entry point is `run_strategy_eval.py`.

### Pipeline

1. **Market data preparation** (`backtester/pipeline.py` →
   `prepare_market_context()`): fetches hourly candles (plus optional datasets
   like funding rates) for all symbols across the evaluation period, including
   warmup bars for indicator computation.

2. **Signal generation**: the evaluator calls
   `generator.generate_backtest_signals(prepared_context, symbols, start, end)`.
   The strategy iterates over the prepared candles and returns `Signal` objects.
   This is the method every strategy **must implement** to be backtestable.

3. **Signal resolution** (`backtester/engine.py` → `backtest_signal()`):
   each signal is resolved into a `TradeResult` using a multi-level fallback:
   agg trades (tick-level) → 1m candles → 1h candles. Market data is fetched
   on demand via `BacktestExecutionSession` with shared disk caches.

4. **Aggregation** (`backtester/evaluator.py` → `StrategyEvaluator`):
   groups windows into contiguous fetch periods for efficiency, runs signal
   generation and resolution in parallel, and produces an `EvaluationReport`
   with per-window and per-category metrics (win rate, PnL, drawdown, profit
   factor).

5. **Lookahead validation** (`backtester/validation.py` →
   `validate_no_lookahead()`): truncates market context to each signal's
   timestamp and re-runs generation. If a signal disappears, it was using
   future data.

### Evaluation Windows

The evaluator uses a coded window calendar instead of ad-hoc date ranges:

- `DEVELOPMENT_WINDOWS` — tune and iterate
- `EVALUATION_WINDOWS` — out-of-sample generalization check
- `ALL_WINDOWS` — full report

### Drawdown Semantics

In the category table output, each row's `DD` is computed from trades in that
category only. The summary `MDD` is computed from the full chronological trade
stream across all selected windows.


## Live Trading

The live engine polls signal generators on their declared interval, converts
signals to exchange orders, and manages positions through their full lifecycle.
The entry point is `live/run.py`.

### Components

**LiveEngine** (`live/engine.py`): the main loop. Each registered generator
gets a `_GeneratorSlot` with its own budget and poll schedule. On each tick the
engine:
1. Checks fills on pending/open positions via the tracker
2. Polls each generator whose interval boundary has passed
3. For each returned signal, checks capital and slot availability, then hands
   it to the executor

**OrderExecutor** (`live/executor.py`): converts a `Signal` into exchange
orders — sets leverage, computes quantity, places the entry order (market or
limit), then places TP/SL bracket orders after fill.

**PositionTracker** (`live/tracker.py`): manages position state
(`PENDING_ENTRY` → `OPEN` → `CLOSED`/`FAILED`), persists state to
`~/.claude_trader/live_state.json` so positions survive restarts, and
reconciles with the exchange on startup.

### Multi-Strategy Support

`LiveEngine` accepts multiple `(generator, GeneratorBudget)` pairs. It
validates that symbol spaces are disjoint (no symbol traded by two generators)
and polls each on its own schedule. For backtesting the same composition, use
`CompositeSignalGenerator` from `live/signal_generator.py`.

### Configuration

Loaded from environment variables or `~/.claude_trader/live_config.json`.
The `--config` flag overrides the default path.

| Variable | Purpose |
|----------|---------|
| `BINANCE_API_KEY` | API key |
| `BINANCE_API_SECRET` | API secret |
| `BINANCE_BASE_URL` | Endpoint (default: `https://fapi.binance.com`) |
| `BINANCE_POSITION_SIZE` | Position size in USDT |
| `BINANCE_MAX_POSITIONS` | Max concurrent positions |
| `BINANCE_ORDER_CHECK_INTERVAL` | Seconds between fill checks |
| `BINANCE_TESTNET` | Use testnet |

Strategy parameters (TP/SL, thresholds, gates) live in strategy code and the
signals they produce — not in live config.


## Writing a Strategy

```python
from live.signal_generator import SignalGenerator
from backtester.models import Signal, PositionType
from backtester.pipeline import PreparedMarketContext

class MyStrategy(SignalGenerator):
    @property
    def symbols(self):
        return ["BTCUSDT", "ETHUSDT"]

    def indicator_request(self):
        return ("rsi_14", "bb_upper", "bb_lower", "atr")

    def generate_backtest_signals(self, prepared_context, symbols, start, end):
        signals = []
        for symbol in symbols:
            # indicator_frame() has precomputed rsi_14, bb_upper, etc.
            df = prepared_context.indicator_frame(symbol)
            mask = (df["close_time"] >= start) & (df["close_time"] < end)
            for _, row in df[mask].iterrows():
                if row["rsi_14"] < 30 and row["close"] < row["bb_lower"]:
                    signals.append(Signal(
                        signal_date=row["close_time"],
                        position_type=PositionType.LONG,
                        ticker=symbol,
                        tp_pct=3.0,
                        sl_pct=1.5,
                    ))
        return signals

    def poll(self):
        # Live: check current market, return Signal or None
        return None
```

**Backtest it:**
```bash
python run_strategy_eval.py --strategy my_strategy.py:MyStrategy --windows development
```

**Deploy it live:**
```python
from live.engine import LiveEngine
from live.models import LiveConfig, GeneratorBudget

engine = LiveEngine(
    generators=[(MyStrategy(), GeneratorBudget(position_size_usdt=100, max_positions=3))],
    config=LiveConfig.load(),
)
engine.start()
```
