---
title: "RandomSearchOptimizer<T, TInput, TOutput>"
description: "Implements random search hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Implements random search hyperparameter optimization.

## How It Works

**For Beginners:** Random search randomly tries different hyperparameter combinations.
While simple, it's surprisingly effective and often outperforms grid search, especially
when some hyperparameters are more important than others.

How it works:

1. Randomly sample hyperparameter values from the search space
2. Train/evaluate the model with those hyperparameters
3. Record the results
4. Repeat for the specified number of trials
5. Return the best configuration found

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomSearchOptimizer(Boolean,Nullable<Int32>)` | Initializes a new instance of the RandomSearchOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration. |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration to try. |

