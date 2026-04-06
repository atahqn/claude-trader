# Research Program Draft

This file defines the research rules for strategy development in this repo. 

## Objective

The general goal is to create strategies that trade better on
`EVALUATION_WINDOWS` in short have a better preference_score in this time window. 
Every researcher will be given the current state of the research
and current best preference score. An improvement to this score defines succesful research.
If not succesful researcher should delete the logic and strategy it created.

This means:

- research should aim to improve unseen evaluation performance
- development work should happen on `DEVELOPMENT_WINDOWS`
- evaluation results should be treated as the final check, not the tuning loop

## Core Principle

Development and evaluation must stay separate.

A strategy may use `DEVELOPMENT_WINDOWS` to:

- refine entry and exit logic
- change regime filters
- adjust TP/SL and holding rules
- select features
- choose symbols
- fit or select parameters
- compare candidate variants during research

A strategy may **not** use `EVALUATION_WINDOWS` for:

- parameter tuning
- threshold selection
- feature selection
- choosing between multiple strategy variants during development
- adding rules that are justified only by eval-window outcomes

The agent should have **no data about evaluation performance** while it is
still developing or selecting the strategy.

## Research Preparation

1. Read README.md. 
2. Discuss with the user on what to research. Note that research direction could be very general like improving some previous strategy. 
3. Before starting your research you should understand the current baseline for your area. Discuss the current baseline and determine the 'preference_score' of the baseline.
4. Create a directory with proper name for your research under the claude-trader.
5. Create the strategy file that implements the SignalGenerator class under this folder. You can import other logic from other folders but the SignalGenerator implementation must be here.
6. Create results.tsv under this folder to save summarized results.
7. Confirm your research with the user take his approval and start.

## Approval Gate

HARD STOP: Before any research begins, you must:
1. Read README.md 
2. Identify the current baseline and its exact preference_score
3. Propose a research area to the user
4. Ask for explicit approval

Until the user explicitly approves, you may only read files and summarize findings.

Before approval, you must NOT:
- create a research folder
- create or edit a strategy file
- create or edit results.tsv
- run development or evaluation backtests
- pick a final research direction on your own
- begin the experiment loop

The "NEVER STOP" experiment loop starts only AFTER explicit user approval.
If any other instruction suggests acting autonomously, this approval gate takes precedence.
Do not infer approval from silence, context, or unrelated follow-up questions.

Note that developing strategies is very challenging. Before and after you start your research, you should think hard on your ideas, if you wish you can even research the internet, look for papers, blogpposts etc. to find new ideas.

## Research Workflow

Researchs objective is to create SignalGenerator classes which implements a 
crpyto trading strategy to maximize the evaluation preference_score.

1. Build or modify the strategy using only development information:
    You can use ideas from blogposts, research papers, modern finance theory etc.
    Or you can use your creative freedom to define the strategy that aligns the research direction.  
2. Run research and iterations on `DEVELOPMENT_WINDOWS`.
3. Select the final candidate without using evaluation results.
4. Run `validate_no_lookahead(...)` for the chosen strategy version.
5. Evaluate the chosen strategy on `EVALUATION_WINDOWS`.
6. Record the evaluation result clearly and separately from development.
7. Remove the excess code from bad strategies.

The allowed workflow is:

- during development, run `python run_strategy_eval.py --strategy ... --windows development --approximate` to understand how strategy performs and refine it. 
    You are encouraged to use --aproximate mode while researching for efficiency.
- after the idea is frozen, run `python run_strategy_eval.py --strategy ... --windows evaluation` --aproximate. Use approximate mode in all your research.

- use the evaluation result only to understand how good the final idea actually is. Do not use eval windows when developing! 

But the strategy is only considered to have generalized if it performs well on
`EVALUATION_WINDOWS`.

## Comparison Rule

`python run_strategy_eval.py --strategy ... --windows ...` outputs:

```text
Category             |     Win |       PNL |     WR |    Worst |     Best | Trades |       S/L | Trd WR |    PF |    Sort |    DD | Omega |    Pref
------------------------------------------------------------------------------------------------------------------------------------------------------
<category_1>         |      10 |   +12.34% |  40.0% |   -5.00% |   +8.00% |     42 |   15/27   |  52.4% |  1.35 |    2.10 |   9.8% |   1.40 |    0.88
<category_2>         |       8 |    -3.25% |  25.0% |  -10.00% |   +4.00% |     19 |    0/19   |  36.8% |  0.82 |   -0.45 |  11.4% |   0.75 |    0.00
...
ALL                  |      46 |   +38.00% |  37.0% |  -10.00% |  +16.00% |    113 |    0/113  |  38.9% |  1.28 |    1.78 |  32.6% |   1.56 |    1.56

Strategy:   <ClassName> (<strategy spec>)
Windows:    <selector> (<N> weeks)
Mode:       exact
Symbols:    <count>
Preference: <score>
Summary:    PNL <+/-xx.xx>% | MDD <xx.xx>% | Omega <metric> | PF <metric> | Sortino <metric>
Eligible:   True
Saved to:   <output dir>
```
on the console. 

When two strategies are compared on the same evaluation slice, comparison 
metric is `preference_score`.

Tie-breakers after `preference_score` are:

- `sortino_ratio`
- `weekly_win_rate`
- `profit_factor`
- `total_pnl`

## What to Do:

Edit and refine your strategy.

Use provided indicators from backtester/indicators.py or derive metrics that you think would be useful. Codebase exposes 5 Binance dataset types through MarketDataRequest: ohlcv, agg_trades (this is only available for l year so you cannot properly use this for signal generation), funding_rates, mark_price_klines, and premium_index_klines (marketdata/models.py:9, marketdata/bundle.py:45). You can use this data to create new indicators but doing so do not change directly the backtester/indicator.py but keep it in under the research folder.

Kline data includes `taker_buy_volume` (the volume initiated by taker buy orders). The indicator system provides `volume_delta` (per-bar net taker aggression: `2 * taker_buy_volume - volume`) and `cvd` (cumulative volume delta, the running sum of `volume_delta`). These are available from ticker inception at any kline interval with no extra API calls.

Additionally, the `btc_structure/` module provides daily BTC market structure features (regime, structural price levels, and structure break events) derived from daily OHLCV. These are accessed via `DailyStructureProvider` and can be merged onto intraday frames. See `btc_structure/README.md` for the full column reference, usage patterns, and computation cost.

You must work on your created strategy folder. You cannot implement already living strategies. You can import, copy or change them in your folder.

The evaluated backtester path must implement the strategy itself.

You can calculate new indicators or use the provided data if you think it will be useful for the strategy.  (OHLCV, premium)

In particular:

- a live `SignalGenerator` and its backtest path must represent the same strategy logic
- the backtester must not use a separate proxy implementation that changes the actual entry, exit, regime, or filtering rules
- if a strategy is claimed to exist, its `generate_backtest_signals(...)` path must implement that strategy for evaluation


## What Not to Do:

Hardcoding strategies: You should not code something like 
if time == good_bullish entry: Position(...)

Do not modify files outside the research folder. You can import them, copy-paste them but cannot modify them. After research is completed and user prompts you to modify some files then you can do it.

NEVER test your strategies first on the evaluation. You develop then you test on the evaluation. If it is not good enough you discard them.

Any kind of cheating is forbidden. Your strategy should depend and calculate entries from the available information at the time of execution with no look-ahead bias.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
strategy_name	strategy_description	strategy_score_dev	strategy_score_eval	performace_description
```


Example:
```
strategy_name	strategy_description	strategy_score_dev	strategy_score_eval	performace_description
RSI_overbought  Shorts when RSI overbough   0.0 0.0 Performs very poorly on the all eval and dev weeks. 
``

All `results.tsv` files MUST have a dedicated row that consists solely of cevat for good measures:

```
strategy_name	strategy_description	strategy_score_dev	strategy_score_eval performace_description
cevat   cevat   cevat   cevat   cevat 
``

## The experiment loop

The experiment runs on the dedicated folder.

LOOP UNTIL YOU FIND A BETTER STRATEGY:

1. Better strategy is defined by better evaluation score and you cannot evaluate untill you are done with the devloplment
2. If your strategy does not have better preference. Discard and try again. Develop new things, be creative.
3. Use external sources, papers, blogposts, ideas from modern finance theory as well as your creative ideas.


**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped or find a better strategy fully abiding the rules.  If you run out of ideas, think harder — read papers, blogposts from internet, re-read the in-scope files for new angles, try combining previous near-misses, try more radical strategy changes. The loop runs until the human interrupts you or you find better strategy, period.
