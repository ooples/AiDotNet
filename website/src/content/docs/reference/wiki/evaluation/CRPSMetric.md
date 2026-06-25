---
title: "CRPSMetric<T>"
description: "Continuous Ranked Probability Score metric for probabilistic predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Probabilistic`

Continuous Ranked Probability Score metric for probabilistic predictions.

## For Beginners

CRPS tells you how good your probability forecasts are.
Unlike log score, CRPS:

- Has the same units as your predicted variable
- For point predictions, equals mean absolute error
- Rewards both accuracy and appropriate uncertainty

Lower CRPS is better.

## How It Works

CRPS is a proper scoring rule that measures the quality of probabilistic predictions.
It generalizes mean absolute error to full probability distributions.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Description` |  |
| `Direction` |  |
| `MaxValue` |  |
| `MinValue` |  |
| `Name` |  |
| `RequiresProbabilities` |  |
| `SupportsMultiClass` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(ReadOnlySpan<>,ReadOnlySpan<>)` |  |
| `ComputeFromDistributions(IParametricDistribution<>[],Vector<>)` | Computes the mean CRPS for probabilistic predictions. |
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` |  |

