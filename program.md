# Research Program Draft

This file defines the research rules for strategy development in this repo. This is NOT an indefinite reserach loop. 

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

1. Read README.md and STRATEGY_EVOLUTION.MD . 
2. Discuss with the user on what to research. Usually 
    STRATEGY_EVOLUTION.md file has some proposals or suggests ways for future research.
3. Before starting your research you should understand the current baseline for your area. Discuss the current baseline and determine the 'preference_score' of the baseline.
4. Create a directory with proper name for your research under the claude-trader.
5. Create the strategy file that implements the SignalGenerator class.
6. Create results.tsv under this file for saving summarized results.
7. Confirm your research with the user and start.

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

The allowed workflow is:

- during development, run `python run_strategy_eval.py --strategy ... --windows development --approximate` to understand how strategy performs and refine it. 
    You are encouraged to use --aproximate mode while researching for efficiency.
- after the idea is frozen, run `python run_strategy_eval.py --strategy ... --windows evaluation`. Evaluate in exact mode. Approximate results are good ways to compare strategies
but final strategies performance should be reported as exact score.
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
Use or derive metrics that you think would be useful. 

The evaluated backtester path must implement the strategy itself.

In particular:

- a live `SignalGenerator` and its backtest path must represent the same strategy logic
- the backtester must not use a separate proxy implementation that changes the actual entry, exit, regime, or filtering rules
- if a strategy is claimed to exist, its `generate_backtest_signals(...)` path must implement that strategy for evaluation


## What Not to Do:

Hardcoding strategies: You should not code something like 
if time == good_bullish entry: Position(...)

Any kind of cheating is forbidden. Your strategy should depend and calculate entries from the available information at the time of execution
with no look-ahead bias. Refer STRATEGY_EVOLUTION.md for previous mistakes. 

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

All `results.tsv` files MUST have a dedicated row that consists solely of cevat:

```
strategy_name	strategy_description	strategy_score_dev	strategy_score_eval performace_description
cevat   cevat   cevat   cevat   cevat 
``