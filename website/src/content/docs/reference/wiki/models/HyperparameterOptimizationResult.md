---
title: "HyperparameterOptimizationResult<T>"
description: "Contains the results of a hyperparameter optimization process."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Contains the results of a hyperparameter optimization process.

## How It Works

**For Beginners:** This stores everything about a hyperparameter search,
including the best hyperparameters found and all the trials that were run.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperparameterOptimizationResult` | Initializes a new instance of the HyperparameterOptimizationResult class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AllTrials` | Gets or sets all trials performed. |
| `BestObjectiveValue` | Gets or sets the best objective value achieved. |
| `BestParameters` | Gets or sets the best hyperparameter values. |
| `BestTrial` | Gets or sets the best trial found during optimization. |
| `CompletedTrials` | Gets or sets the number of completed trials. |
| `EndTime` | Gets or sets the end time of optimization. |
| `FailedTrials` | Gets or sets the number of failed trials. |
| `PrunedTrials` | Gets or sets the number of pruned trials. |
| `SearchSpace` | Gets or sets the search space that was used. |
| `StartTime` | Gets or sets the start time of optimization. |
| `TotalTime` | Gets or sets the total optimization time. |
| `TotalTrials` | Gets or sets the total number of trials run. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptimizationHistory` | Gets the optimization history as a list of (trial number, objective value) pairs. |
| `GetTopTrials(Int32,Boolean)` | Gets the top N trials by objective value. |

