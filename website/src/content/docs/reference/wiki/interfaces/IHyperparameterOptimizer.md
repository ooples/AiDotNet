---
title: "IHyperparameterOptimizer<T, TInput, TOutput>"
description: "Defines the contract for hyperparameter optimization algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for hyperparameter optimization algorithms.

## How It Works

A hyperparameter optimizer automatically searches for the best hyperparameters for a machine learning model
by trying different combinations and evaluating their performance.

**For Beginners:** Think of hyperparameters as the "settings" for your machine learning algorithm
(like learning rate, number of layers, etc.). A hyperparameter optimizer is like an automatic tuner that
tries different settings to find the combination that works best for your data.

Common optimization strategies include:

- Grid Search: Tries every possible combination in a predefined grid
- Random Search: Randomly samples combinations
- Bayesian Optimization: Uses past results to intelligently choose what to try next
- Hyperband: Efficiently allocates resources to promising configurations

Why hyperparameter optimization matters:

- Manual tuning is time-consuming and error-prone
- Good hyperparameters can dramatically improve model performance
- Systematic search ensures you don't miss good configurations
- Enables reproducible model selection

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAllTrials` | Gets all trials performed during optimization. |
| `GetBestTrial` | Gets the best trial from the optimization. |
| `GetTrials(Func<HyperparameterTrial<>,Boolean>)` | Gets trials that meet a certain criteria. |
| `Optimize(Func<Dictionary<String,Object>,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameter configuration. |
| `OptimizeModel(IModel<,,>,ValueTuple<,>,ValueTuple<,>,HyperparameterSearchSpace,Int32)` | Searches for the best hyperparameters for a specific model. |
| `ReportTrial(HyperparameterTrial<>,)` | Reports the result of a trial. |
| `ShouldPrune(HyperparameterTrial<>,Int32,)` | Determines if a trial should be pruned (stopped early) to save resources. |
| `SuggestNext(HyperparameterTrial<>)` | Suggests the next hyperparameter configuration to try based on past trials. |

