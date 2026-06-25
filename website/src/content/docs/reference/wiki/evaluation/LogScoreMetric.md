---
title: "LogScoreMetric<T>"
description: "Logarithmic scoring metric for probabilistic predictions (Negative Log Likelihood)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Probabilistic`

Logarithmic scoring metric for probabilistic predictions (Negative Log Likelihood).

## For Beginners

This metric measures how well your predicted probability
distributions match reality. A lower score means better predictions.
It's particularly useful for models that output full distributions rather
than just point predictions.

## How It Works

Computes the mean negative log likelihood across all predictions.
This is the most common metric for evaluating probabilistic forecasts.

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
| `ComputeFromDistributions(IParametricDistribution<>[],Vector<>)` | Computes the mean log score for probabilistic predictions. |
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` |  |

