---
title: "HyperparameterOptimizer<T>"
description: "Provides hyperparameter optimization for Gaussian Processes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Provides hyperparameter optimization for Gaussian Processes.

## For Beginners

Gaussian Processes have hyperparameters (like kernel length scale,
signal variance, and noise variance) that greatly affect performance. This class helps
find good values for these hyperparameters automatically.

Methods available:

- Grid Search: Try all combinations from a predefined grid
- Random Search: Try random combinations (often more efficient than grid)
- Gradient Descent: Follow gradients of log marginal likelihood
- Bayesian Optimization: Use a GP to model the objective (meta!)

The optimization target is typically the log marginal likelihood (LML):

- Higher LML = better fit to data (accounting for model complexity)
- LML naturally balances fit quality with model simplicity

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperparameterOptimizer(Int32,Double,Int32)` | Initializes a new hyperparameter optimizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CrossValidate(Matrix<>,Vector<>,Func<Dictionary<String,Double>,ValueTuple<Func<Matrix<>,Vector<>,ValueTuple<Vector<>,Vector<>>>,Action<Matrix<>,Vector<>>>>,Dictionary<String,Double>,Int32)` | Performs cross-validation to estimate generalization performance. |
| `GenerateGridCombinations(List<String>,List<Double[]>)` | Generates all combinations from a parameter grid. |
| `GradientDescent(Dictionary<String,Double>,Func<Dictionary<String,Double>,ValueTuple<Double,Dictionary<String,Double>>>,Double,Dictionary<String,ValueTuple<Double,Double>>)` | Performs gradient-based optimization of hyperparameters. |
| `GridSearch(Dictionary<String,Double[]>,Func<Dictionary<String,Double>,Double>)` | Performs grid search over hyperparameter values. |
| `MultiStartOptimization(Dictionary<String,ValueTuple<Double,Double>>,Func<Dictionary<String,Double>,ValueTuple<Double,Dictionary<String,Double>>>,Int32,Double)` | Performs multi-start optimization with random initializations. |
| `RandomSearch(Dictionary<String,ValueTuple<Double,Double>>,Func<Dictionary<String,Double>,Double>,Int32,String[])` | Performs random search over hyperparameter ranges. |

