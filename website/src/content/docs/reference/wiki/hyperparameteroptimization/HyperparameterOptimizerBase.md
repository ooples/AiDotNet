---
title: "HyperparameterOptimizerBase<T, TInput, TOutput>"
description: "Base class for hyperparameter optimization algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.HyperparameterOptimization`

Base class for hyperparameter optimization algorithms.

## How It Works

**For Beginners:** This abstract base class provides common functionality for hyperparameter
optimization. It handles trial management, result tracking, and provides helper methods for
finding best trials while leaving the specific optimization strategy to derived classes.

Key features:

- Thread-safe trial management
- Consistent result aggregation
- Helper methods for finding best trials
- Support for both maximization and minimization objectives

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperparameterOptimizerBase(Boolean)` | Initializes a new instance of the HyperparameterOptimizerBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateOptimizationResult(HyperparameterSearchSpace,DateTime,DateTime,Int32)` | Creates an optimization result from the current trials. |
| `EvaluateTrialSafely(HyperparameterTrial<>,Func<Dictionary<String,Object>,>,Dictionary<String,Object>)` | Evaluates a trial with the objective function and handles exceptions. |
| `FindBestTrial(List<HyperparameterTrial<>>)` | Finds the best trial from a list of completed trials. |
| `GetAllTrials` | Gets all trials performed during optimization. |
| `GetBestTrial` | Gets the best trial from the optimization. |
| `GetTrials(Func<HyperparameterTrial<>,Boolean>)` | Gets trials that meet a certain criteria. |
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration. |
| `OptimizeModel(IModel<,,>,ValueTuple<,>,ValueTuple<,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameters for a specific model. |
| `ReportTrial(HyperparameterTrial<>,)` | Reports the result of a trial. |
| `ShouldPrune(HyperparameterTrial<>,Int32,)` | Determines if a trial should be pruned (stopped early). |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration to try. |
| `ValidateOptimizationInputs(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Validates the optimization inputs. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Maximize` | Whether to maximize (true) or minimize (false) the objective function. |
| `SearchSpace` | The search space being optimized. |
| `SyncLock` | Lock object for thread-safe operations. |
| `Trials` | Collection of all trials performed during optimization. |

