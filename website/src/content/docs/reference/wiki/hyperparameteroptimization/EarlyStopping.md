---
title: "EarlyStopping<T>"
description: "Provides early stopping functionality for hyperparameter optimization and training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Provides early stopping functionality for hyperparameter optimization and training.

## How It Works

**For Beginners:** Early stopping is a technique to:

- Stop training when performance stops improving
- Prevent overfitting by not training too long
- Save compute resources by terminating hopeless trials

Key concepts:

- Patience: How many checks without improvement before stopping
- Min Delta: Minimum improvement to count as "better"
- Best Value: The best score seen so far
- Counter: Tracks consecutive non-improvements

Early stopping is essential for efficient hyperparameter search because
it allows quick termination of poor configurations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EarlyStopping(Int32,Double,Boolean,EarlyStoppingMode)` | Initializes a new instance of the EarlyStopping class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BestEpoch` | Gets the epoch at which the best value was observed. |
| `BestValue` | Gets the best value observed. |
| `EpochsSinceBest` | Gets the number of epochs since the best value was observed. |
| `History` | Gets the history of values that were checked. |
| `ShouldStop` | Gets whether early stopping has been triggered. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Check(,Int32)` | Checks if training should stop based on the new value. |
| `Check(Double,Int32)` | Checks if training should stop based on the new value. |
| `GetState` | Gets a summary of the early stopping state. |
| `IsImprovement(Double)` | Determines if the new value is an improvement over the best. |
| `Reset` | Resets the early stopping state. |

