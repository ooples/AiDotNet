---
title: "UnsupervisedAutoMLBase<T>"
description: "Base class for unsupervised AutoML search strategies (e.g., clustering AutoML, grid search)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AutoML`

Base class for unsupervised AutoML search strategies (e.g., clustering AutoML, grid search).

## For Beginners

This is the foundation for tools that automatically find the best
clustering algorithm and parameters for your data — without needing labeled examples.

## How It Works

Unlike supervised AutoML (which extends `AutoMLModelBase`),
unsupervised AutoML operates on unlabeled data. This base class provides common infrastructure
for unsupervised search strategies: trial tracking, metric optimization, search space management,
and early stopping.

## Properties

| Property | Summary |
|:-----|:--------|
| `BestScore` | Gets the best score achieved during the search. |
| `Engine` | Hardware-accelerated engine for vector/tensor operations. |
| `HigherIsBetter` | Gets whether higher metric values are better (derived from the metric type). |
| `MaxTrials` | Gets or sets the maximum number of trials to run. |
| `PrimaryMetric` | Gets the primary metric name for evaluation dictionary lookups. |
| `PrimaryMetricType` | Gets or sets the primary evaluation metric type. |
| `TimeLimit` | Gets or sets the time limit for the search. |
| `TrialsEvaluated` | Gets the total number of trials evaluated. |

## Methods

| Method | Summary |
|:-----|:--------|
| `IsBetterScore(Double)` | Determines whether a new score is better than the current best. |
| `TryUpdateBestScore(Double)` | Updates the best score if the new score is better. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for the specified type T. |

