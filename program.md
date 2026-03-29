# Research Program Draft

This file defines the research rules for strategy development in this repo.
Its purpose is to keep strategy research honest and to optimize for **better
performance on unseen evaluation windows**, not just better in-sample results.

## Objective

The goal is to create strategies that trade better on
`EVALUATION_WINDOWS`.

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

## Required Workflow

1. Build or modify the strategy using only development information.
2. Run research and iteration on `DEVELOPMENT_WINDOWS`.
3. Select the final candidate without using evaluation results.
4. Run `validate_no_lookahead(...)` for the chosen strategy version.
5. Evaluate the chosen strategy on `EVALUATION_WINDOWS`.
6. Record the evaluation result clearly and separately from development.

## Interpretation Rule

Evaluation is the out-of-sample test.

Development performance is useful for:

- debugging
- shaping the strategy
- rejecting obviously weak ideas
- comparing internal candidate variants before final selection

But the strategy is only considered to have generalized if it performs well on
`EVALUATION_WINDOWS`.

## Comparison Rule

When two strategies are compared on the same evaluation slice, the default
comparison metric is `preference_score`.

Tie-breakers after `preference_score` are:

- `sortino_ratio`
- `weekly_win_rate`
- `profit_factor`
- `total_pnl`

## Fixed Evaluation Harness

The researcher must not edit the evaluation runner or the coded evaluation
windows as part of strategy research.

In this repo that means:

- do not modify [run_strategy_eval.py](/home/caner/claude-trader/run_strategy_eval.py) in order to make a strategy look better
- do not modify [eval_windows.py](/home/caner/claude-trader/backtester/eval_windows.py) in order to make a strategy look better

The allowed workflow is:

- during development, run `python run_strategy_eval.py --strategy ... --windows development`
- after the idea is frozen, run `python run_strategy_eval.py --strategy ... --windows evaluation`
- use the evaluation result only to understand how good the final idea actually is

## Practical Standard For Agents

An agent working in this repo should behave as follows:

- assume `DEVELOPMENT_WINDOWS` is the only place for iteration
- avoid looking at eval results until the strategy version is frozen
- treat eval as a one-way check on the chosen candidate
- do not edit the evaluation runner or the window definitions during strategy research
- document clearly whether a result is development, evaluation, or all-window
- avoid one-off evaluation scripts when shared evaluator flow already exists

## Canonical Tools

Use the shared research path:

- `backtester.StrategyEvaluator`
- `backtester.DEVELOPMENT_WINDOWS`
- `backtester.EVALUATION_WINDOWS`
- `backtester.ALL_WINDOWS`
- `backtester.validate_no_lookahead(...)`

## Status

This is a draft. It should be refined until it is strict enough to guide future
agents unambiguously, but not so rigid that it blocks legitimate new strategy
research.
