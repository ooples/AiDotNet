---
title: "GridSearchOptimizer<T, TInput, TOutput>"
description: "Implements grid search hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Implements grid search hyperparameter optimization.

## How It Works

**For Beginners:** Grid search systematically tries every possible combination
of hyperparameters from a predefined grid.

How it works:

1. Define a grid of values for each hyperparameter
2. Generate all possible combinations
3. Try each combination in sequence
4. Return the best configuration found

Advantages:

- Guaranteed to find the best combination in the grid
- Systematic and reproducible

Disadvantages:

- Can be very slow with many hyperparameters (combinatorial explosion)
- Wastes time on unpromising regions
- Only searches discrete values, not continuous ranges

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GridSearchOptimizer(Boolean)` | Initializes a new instance of the GridSearchOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration using grid search. |
| `SuggestNext(HyperparameterTrial<>)` | Grid search doesn't use suggestions - it pre-generates all combinations. |

