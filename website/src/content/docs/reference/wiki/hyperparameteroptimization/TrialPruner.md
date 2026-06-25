---
title: "TrialPruner<T>"
description: "Provides trial pruning functionality for hyperparameter optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.HyperparameterOptimization`

Provides trial pruning functionality for hyperparameter optimization.

## How It Works

**For Beginners:** Trial pruning allows early termination of unpromising trials:

- During training, periodically report intermediate results
- Pruner compares against historical data from other trials
- If the current trial is clearly worse, it gets pruned (stopped early)
- This saves computational resources for more promising trials

Key concepts:

- Intermediate Value: A performance metric reported during training
- Step: The training progress when the value was reported
- Pruning: Terminating a trial that's unlikely to improve

Trial pruning is especially useful with Bayesian optimization and Hyperband
where many configurations are evaluated.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrialPruner(Boolean,PruningStrategy,Double,Int32,Int32)` | Initializes a new instance of the TrialPruner class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckThreshold(Double,Double)` | Checks if a trial should be pruned based on a threshold. |
| `GetPercentile(List<Double>,Double)` | Gets the percentile value from a list. |
| `GetStatistics` | Gets statistics about pruning decisions. |
| `GetValuesAtStep(String,Int32)` | Gets values from other trials at the same step. |
| `MarkComplete(String)` | Marks a trial as complete (called when trial finishes without pruning). |
| `MedianPruningCheck(String,Int32,Double)` | Median pruning: prune if below median of completed trials at this step. |
| `PercentilePruningCheck(String,Int32,Double)` | Percentile pruning: prune if in bottom percentile. |
| `ReportAndCheckPrune(HyperparameterTrial<>,Int32,)` | Reports an intermediate value and checks if the trial should be pruned. |
| `ReportAndCheckPrune(String,Int32,Double)` | Reports an intermediate value and checks if the trial should be pruned. |
| `Reset` | Resets the pruner state. |
| `ShouldPrune(String,Int32,Double)` | Determines if a trial should be pruned based on its intermediate value. |
| `SuccessiveHalvingCheck(String,Int32,Double)` | Successive halving check: compare with top half of trials. |

