---
title: "FinancialAutoML<T>"
description: "AutoML implementation for finance models (forecasting and risk)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.AutoML`

AutoML implementation for finance models (forecasting and risk).

## For Beginners

This class is a "model picker" for finance tasks.
It tries several finance models and chooses the one that scores best on your data.

## How It Works

FinancialAutoML searches across a curated set of finance models while preserving
the facade pattern. You provide the architecture and budget; AutoML selects the model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinancialAutoML(FinancialAutoMLOptions<>,Random)` | Initializes a new FinancialAutoML instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBudget(AutoMLBudgetOptions)` | Applies the AutoML budget to time and trial limits. |
| `CreateInstanceForCopy` | Creates a new instance for cloning. |
| `CreateModelAsync(Type,Dictionary<String,Object>)` | Creates a finance model for the given trial parameters. |
| `EnsureDefaultCandidateModels` | Ensures a default candidate model list is available. |
| `EnsureDefaultOptimizationMetric` | Ensures a default optimization metric is selected when none is specified. |
| `GetDefaultModelsForDomain(FinancialDomain)` | Gets the default candidate models for a finance domain. |
| `GetDefaultSearchSpace(Type)` | Gets the default search space for a model type. |
| `IsHigherBetter(MetricType)` | Determines whether a metric should be maximized or minimized. |
| `SearchAsync(Tensor<>,Tensor<>,Tensor<>,Tensor<>,TimeSpan,CancellationToken)` | Runs the AutoML search loop. |
| `SuggestNextTrialAsync` | Suggests the next trial parameters. |

