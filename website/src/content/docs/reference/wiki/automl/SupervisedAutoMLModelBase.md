---
title: "SupervisedAutoMLModelBase<T, TInput, TOutput>"
description: "Base class for AutoML implementations that train and score supervised models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AutoML`

Base class for AutoML implementations that train and score supervised models.

## For Beginners

AutoML is an automatic "model picker + tuner".
A supervised AutoML run:

- Tries a candidate model configuration (a "trial").
- Trains it on your training data.
- Scores it on validation data using a metric (like RMSE or Accuracy).
- Repeats until it finds a strong model or runs out of budget.

Concrete strategies (random search, Bayesian optimization, etc.) decide how to pick the next trial.

## How It Works

This base class provides common trial execution logic (create model, train, evaluate, record results)
for AutoML strategies that operate on supervised learning datasets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SupervisedAutoMLModelBase(Random)` | Initializes a new supervised AutoML model with sensible default dependencies. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BudgetPreset` | Gets or sets the compute budget preset used to choose sensible built-in defaults. |
| `CrossValidationOptions` | Gets or sets cross-validation options for trial evaluation. |
| `EnsembleOptions` | Gets or sets options controlling optional post-search ensembling. |
| `Random` | Gets the RNG used for sampling candidate trials. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateSubset(,,Int32[])` | Creates a subset of the input/output data based on the given row indices. |
| `EnsureDefaultOptimizationMetric()` | Applies an industry-default metric if the user didn't explicitly choose one. |
| `ExecuteTrialAsync(Type,Dictionary<String,Object>,,,,,CancellationToken)` | Runs a single trial (create, train, evaluate, record history). |
| `ExecuteTrialWithCrossValidationAsync(Type,Dictionary<String,Object>,,,CancellationToken)` | Executes a trial using k-fold cross-validation for more robust evaluation. |
| `GetRowCount()` | Gets the row count from the input data. |
| `PickCandidateModelType` | Picks a model type uniformly from the configured candidate list. |
| `TrySelectEnsembleAsBestAsync(,,,,DateTime,CancellationToken)` | Attempts to build and select an ensemble as the final model based on `EnsembleOptions`. |

