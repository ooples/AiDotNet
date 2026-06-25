---
title: "CrossValidationEngine<T>"
description: "Engine for executing cross-validation with various strategies and aggregating results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for executing cross-validation with various strategies and aggregating results.

## For Beginners

This engine automates the cross-validation process:

- Splits your data according to the chosen strategy
- Trains your model on each training fold
- Evaluates on each validation fold
- Aggregates results with confidence intervals

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossValidationEngine(CrossValidationOptions)` | Initializes the cross-validation engine. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Execute(ICrossValidationStrategy<>,[0:,0:],[],Func<[0:,0:],[],>,Func<,[0:,0:],[]>,Boolean)` | Performs cross-validation using the specified strategy and model training function. |
| `ExecuteTimeSeries(ICrossValidationStrategy<>,[],Int32,Func<[],>,Func<,[],[]>,Int32)` | Performs cross-validation for time series data. |

