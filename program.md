# Research Program

This file defines the research rules for strategy development in this repo.

## Objective

The general goal is to create strategies that trade better on
`EVALUATION_WINDOWS` — in short, have a better `preference_score` in this time window.

This means:

- research should aim to improve unseen evaluation performance
- development work should happen on `DEVELOPMENT_WINDOWS` (58 windows)
- evaluation results should be treated as the final check, not the tuning loop

**ALL research runs must use `--approximate`.** The default in `run_strategy_eval.py` is exact mode, which is slower and wrong for research iteration. Always pass `--approximate`.

## Preference Score

`preference_score = coverage_penalty * omega_component * drawdown_component`

- `omega_component = gross_positive_weeks / max(gross_negative_weeks, 1.0)`
- `drawdown_component = total_pnl / max(max_drawdown, 5.0%)` — **5% MDD floor**: reducing drawdown below 5% gives no additional benefit
- `coverage_penalty = sqrt(active_weeks / total_weeks) * min(1.0, resolved_trades / 80)` — penalizes low trade volume and inactive weeks

**Eligibility gates** (ineligible = score 0):
- `total_pnl > 0` and `profit_factor > 1.0`
- `resolved_trades >= min(40, max(10, 2 * total_weeks))`
- `active_weeks >= min(8, total_weeks)`

Practical: aim for **80+ resolved trades** to avoid the trade-count penalty. A strategy with 0 or negative PnL always scores 0.

Tie-breakers after `preference_score`: `sortino_ratio` > `weekly_win_rate` > `profit_factor` > `total_pnl`.

---

## Development / Evaluation Separation

This is the single most important rule.

**All tuning, parameter selection, feature selection, and candidate comparison must happen on `DEVELOPMENT_WINDOWS` only.**

`EVALUATION_WINDOWS` (41 windows) are a one-time promotion gate for a frozen candidate. They must **never** be used to:
- tune parameters or thresholds
- select features or symbols
- choose between strategy variants
- add rules justified only by eval-window outcomes
- run multiple near-identical variants as a parameter search

After seeing evaluation results: either accept the candidate as genuinely generalized, or discard it and return to development with a materially new idea. Do not make small parameter tweaks justified by specific evaluation outcomes.

---

## Research Preparation

1. Read `README.md`.
2. Discuss with the user what to research. If new research, do not read existing research folders or prior `results.tsv`. If continuing, read only the relevant prior research specified by the user.
3. Before starting research, understand the related parts of the code.
4. Confirm the research direction with the user, ask whether there is a `preference_score` to beat, and get explicit approval to start. If no target score is provided, the research loop runs indefinitely until the user interrupts it.

Until the user explicitly approves, you may only read files and summarize findings. Before approval, you must NOT create research folders, strategy files, results.tsv, run backtests, pick a research direction, or begin the experiment loop. The "NEVER STOP" experiment loop starts only AFTER explicit user approval. Do not infer approval from silence, context, or unrelated follow-up questions.

After approval:

5. Create a directory with a proper name for the research under `claude-trader`.
6. Create the strategy file implementing the `SignalGenerator` class under this folder. You can import other logic from other folders, but the `SignalGenerator` implementation must live here.
7. Create `results.tsv` under this folder.
8. Start the experiment loop.

Developing strategies is very challenging. Think hard on your ideas before and during research. You can research the internet, look for papers, blogposts etc. to find new ideas.

## Required References

- `README.md` for the overall framework, backtesting flow, and the `SignalGenerator` overview
- `marketdata/README.md` for available datasets, indicators, and key levels
- `btc_structure/README.md` for BTC structure features when that subsystem is relevant

Before writing a strategy, read the references that are relevant to your idea.

### SignalGenerator (`live/signal_generator.py`)

Every strategy extends `SignalGenerator`. Core interface:

| Property / Method | Purpose |
|---|---|
| `symbols` (property, required) | List of tickers this strategy trades |
| `indicator_request()` | Tuple of indicator names to precompute (e.g. `("rsi_14", "atr_14", "bb_upper")`) |
| `market_data_request()` | Declare datasets needed (default: OHLCV only). Opt into funding rates, mark/premium klines, key levels |
| `generate_backtest_signals(ctx, symbols, start, end)` | **Must implement.** Return `list[Signal]` with `signal_date` in `[start, end)` |
| `poll()` | Live trading entry point. Return `Signal`, `list[Signal]`, or `None` |
| `analysis_interval` | Candle timeframe (default `"1h"`) |
| `cooldown_hours` | Min hours between signals on the same symbol (default `0.0`) |
| `required_warmup_bars` | Extra bars before signal window for indicator history (default `100`). Auto-computed from `indicator_request()` when declared. |

Calibration hooks (opt-in): `needs_calibration`, `calibration_interval_hours`, `calibration_lookback_hours`, `param_space()`, `score_params()`, `build_calibration_frame()`, `prepare_score_context()`, `active_params`.

### Signal (`backtester/models.py`)

The contract between strategy and execution:

| Field | Type | Default | Notes |
|---|---|---|---|
| `signal_date` | `datetime` | required | When the signal fires |
| `position_type` | `PositionType` | required | `LONG` or `SHORT` |
| `ticker` | `str` | required | e.g. `"BTCUSDT"` |
| `tp_pct` / `tp_price` | `float \| None` | `None` | At least one required |
| `sl_pct` / `sl_price` | `float \| None` | `None` | At least one required |
| `leverage` | `float` | `1.0` | |
| `entry_price` | `float \| None` | `None` | `None` = market order |
| `entry_delay_seconds` | `int \| None` | `None` | Delay before entry |
| `fill_timeout_seconds` | `int` | `3600` | Max wait for fill |
| `max_holding_hours` | `int` | `72` | Must be positive |
| `size_multiplier` | `float` | `1.0` | Position sizing weight |
| `metadata` | `dict` | `{}` | Arbitrary strategy metadata |

### PreparedMarketContext

The `ctx` object passed to `generate_backtest_signals`:

| Method | Returns | Notes |
|---|---|---|
| `ctx.indicator_frame(symbol)` | `DataFrame` with `close_time` + indicator columns | Primary access when using `indicator_request()` |
| `ctx.slice_analysis_candles(symbol, start, end)` | `list[Candle]` | Raw candle access (open_time, close_time, OHLCV) |
| `ctx.get_key_levels(symbol, t)` | `KeyLevels \| None` | Needs `include_key_levels=True` in `market_data_request()` |
| `ctx.for_symbol(symbol)` | `SymbolMarketContext` | Raw frame access via `.frame` |
| `ctx.data_range(symbol)` | `(first_open, last_end) \| None` | Data availability bounds |

### Available Indicators (`backtester/indicators.py`)

Requested via `indicator_request()`. Dependencies auto-resolved. Use `required_warmup(["..."])` for leading bar count.

| Indicator | Description |
|---|---|
| `rsi_14` | 14-period RSI (0-100) |
| `atr_14` | 14-period ATR |
| `atr_72_avg` | 72-bar rolling mean of ATR |
| `atr_ratio` | `atr_14 / atr_72_avg` (volatility expansion/contraction) |
| `ret_24h`, `ret_48h`, `ret_72h` | N-bar percent returns |
| `vol_sma_20` | 20-bar volume SMA |
| `vol_ratio` | `volume / vol_sma_20` |
| `ema_20` | 20-period EMA of close |
| `bb_upper`, `bb_lower` | Bollinger Bands (20-period SMA +/- 2 std) |
| `bb_pct_b` | %B position within Bollinger Bands (0=lower, 1=upper) |
| `bb_width` | Band width normalized by middle band |
| `kc_upper`, `kc_lower` | Keltner Channel (EMA-20 +/- 1.5 ATR) |
| `squeeze_on` | `True` when Bollinger Bands inside Keltner Channels |
| `squeeze_count` | Consecutive bars in squeeze (resets on release) |
| `mom_slope` | 20-bar linear regression slope of close |
| `body` | Candle body: `close - open` |
| `body_ratio` | `body / (high - low)` |
| `adx_14` | 14-period ADX (trend strength, 0-100) |
| `volume_delta` | Net taker aggression: `2 * taker_buy_volume - volume` |
| `cvd` | Cumulative volume delta (running sum of `volume_delta`) |
| `t3` | Tilson T3 moving average (period 5, volume factor 0.7) |

### Available Datasets

OHLCV columns (`open_time`, `close_time`, `open`, `high`, `low`, `close`, `volume`, `taker_buy_volume`) are always included. Opt-in extras via `MarketDataRequest`:

| Dataset | Key columns merged onto OHLCV frame | Notes |
|---|---|---|
| `DataRequirement.FUNDING_RATES` | `funding_rate`, `funding_mark_price` | 8h frequency, merged asof |
| `DataRequirement.MARK_PRICE_KLINES` | `mark_open`, `mark_high`, `mark_low`, `mark_close` | Same interval as OHLCV |
| `DataRequirement.PREMIUM_INDEX_KLINES` | `premium_open`, `premium_high`, `premium_low`, `premium_close` | Same interval as OHLCV |
| Key levels (`include_key_levels=True`) | PDH/PDL, weekly/monthly open/high/low, session levels | Lookahead-free. Access via `ctx.get_key_levels(symbol, t)` |

```python
from marketdata.models import MarketDataRequest, DataRequirement

# OHLCV only (default)
MarketDataRequest.ohlcv_only()

# OHLCV + funding + key levels
MarketDataRequest(
    datasets=frozenset({DataRequirement.OHLCV, DataRequirement.FUNDING_RATES}),
    include_key_levels=True,
)
```

### BTC Structure (`btc_structure/`)

Daily BTC market structure features from `DailyStructureProvider`. Three column groups:

| Group | Cols | Key columns | Use |
|---|---|---|---|
| `STRUCTURE_REGIME` | 7 | `market_bias_after_close`, `global_continuation_long_flag`, `major_global_bullish_confluence_flag` | Macro directional gating |
| `STRUCTURE_LEVELS` | 10 | Fib retracements (50%, 61.8%, 66%, 34%), 1.618 extensions, pullback flags | Entry zones / targets |
| `STRUCTURE_EVENTS` | 6 | Change of character, break of structure, swing confirmation flags | Bar-level pattern events |

Request explicit columns instead of the full ~510-column matrix unless you have a clear reason.

---

## Research Workflow

```
DEVELOP  -> python run_strategy_eval.py --strategy <path>:<Class> --windows development --approximate
            Iterate, diagnose, refine using development diagnostics.

FREEZE   -> When confident the candidate should generalize, freeze it.
            Record: strategy version, core hypothesis, symbols, parameters, why it should generalize.

VALIDATE -> python run_strategy_validate.py --strategy <path>:<Class> --windows development
            Only proceed to evaluation if validation passes.

EVALUATE -> python run_strategy_eval.py --strategy <path>:<Class> --windows evaluation --approximate
            Exactly one evaluation run per frozen candidate. Do not iterate on eval results.

LOG      -> Record result in results.tsv. Start a new DEVELOP cycle.
```

Use development diagnostics during DEVELOP to understand weaknesses:

- Allowed feedback: `preference_score`, total PnL, max drawdown, omega, profit factor, sortino, weekly win rate, category-level summaries, trade counts, long/short balance, saved trade results
- Diagnose: too few trades, excessive drawdown, weak shorts, poor risk/reward, clustered entries, weak categories
- Do not inspect evaluation trades or breakdowns for iterative diagnosis

## What to Do

Use provided indicators from `backtester/indicators.py` or derive custom indicators under the research folder. Do not modify `backtester/indicators.py` directly.

You must work on your created strategy folder. You can import, copy or change code from other folders into yours.

The backtester path must implement the actual strategy:

- a `SignalGenerator` and its backtest path must represent the same strategy logic
- the backtester must not use a separate proxy that changes the entry, exit, regime, or filtering rules
- `generate_backtest_signals(ctx, symbols, start, end)` must only emit signals with `signal_date` in `[start, end)`

Data guidance:

- default to OHLCV plus `indicator_request()` first
- funding rates, mark-price klines, and premium-index klines are opt-in; only request them when the strategy hypothesis needs them
- key levels are already lookahead-free and are safe to use when needed
- agg trades are not suitable for signal generation
- for BTC structure, request explicit columns instead of the full ~510-column matrix unless there is a clear reason not to

## What Not to Do

Do not hardcode strategies — e.g. `if time == good_bullish_entry: Position(...)`. Any form of cheating is forbidden. Your strategy must calculate entries from the available information at the time of execution with no look-ahead bias.

Do not modify files outside the research folder. You can import them or copy-paste them but cannot modify them. After research is completed and user prompts you to modify files, then you can.

Do not test strategies first on evaluation. Develop first, then test the frozen candidate. Do not run multiple near-identical variants on `EVALUATION_WINDOWS` — evaluation is a promotion gate, not a parameter search.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

Do not delete failed strategies. If a candidate is rejected, keep a clear row describing what it tried and why it failed.

The TSV has a header row and 5 columns:

```
strategy_name	strategy_description	strategy_score_dev	strategy_score_eval	performace_description
```

If a candidate was rejected before evaluation, `strategy_score_eval` may be `-`.

Example:
```
strategy_name	strategy_description	strategy_score_dev	strategy_score_eval	performace_description
RSI_overbought	Shorts when RSI overbought	0.0	0.0	Performs very poorly on all eval and dev weeks.
```

All `results.tsv` files MUST include a dedicated cevat row for good luck. It is tradition and must not be skipped:
```
cevat	cevat	cevat	cevat	cevat
```

## The Experiment Loop

The experiment loop is an infinite sequence of sealed research cycles, each following: DEVELOP > FREEZE > VALIDATE > EVALUATE > LOG > repeat.

Within each cycle:

1. Develop only on `DEVELOPMENT_WINDOWS`.
2. Use development results, code understanding, and external research to refine.
3. When confident, freeze the candidate.
4. Before evaluation, record: strategy version, core hypothesis, chosen symbols, parameters, and why it should generalize.
5. Run `python run_strategy_validate.py --strategy ... --windows development` on the frozen candidate.
6. Run exactly one evaluation on `EVALUATION_WINDOWS`.
7. Log the result to `results.tsv`.

Loop rules:

- after seeing evaluation results, do not make small parameter changes justified by specific evaluation outcomes
- if a candidate fails, discard it and return to development with a materially new idea
- preserve failed candidate explanations in `results.tsv` so the same bad idea is not retried
- if a candidate succeeds, record it as a good generalized strategy and continue researching through a new sealed cycle
- use external sources, papers, blogposts, ideas freely during development

**NEVER STOP**: Once the experiment loop has begun (after approval and initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". If no target score was provided, the loop runs indefinitely until the human interrupts you. If a target score was provided, the loop runs until the human interrupts you or until you beat that target.

## Evaluation Windows

The evaluator uses a coded window calendar (`backtester/eval_windows.py`):

| Alias | Windows | Purpose |
|---|---|---|
| `development` | 58 | Tune and iterate |
| `evaluation` | 41 | Frozen candidate promotion gate |
| `all` | 99 | Dev + eval combined |

Targeted development subsets for diagnosis (same `--windows` flag):

| Alias | Windows | Regime focus |
|---|---|---|
| `development_stress` | 10 | High-volatility, drawdown periods |
| `development_bull` | 4 | Bullish regime |
| `development_pairs` | 8 | Paired regime windows |
| `development_random` | 12 | Randomly selected |

## Output Structure

Results are saved to `outputs/strategy_eval/<strategy>_<windows>_<mode>_<timestamp>/`:

| File | Contents |
|---|---|
| `trades.csv` | Every trade: entry/exit times, prices, PnL, exit reason, resolution level |
| `meta.json` | Run config, summary statistics, strategy metadata |
| `category_summary.csv` | Per-category breakdowns of all metrics |

Console output columns: `Category | Win | PNL | WR | Worst | Best | Trades | S/L | Trd WR | PF | Sort | DD | Omega | Pref`
