---
title: "LogLossMetric<T>"
description: "Computes Log Loss (Cross-Entropy Loss): a probabilistic measure that penalizes confident wrong predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Log Loss (Cross-Entropy Loss): a probabilistic measure that penalizes confident wrong predictions.

## For Beginners

Log loss measures how well predicted probabilities match actual outcomes.
Unlike accuracy which just checks if the prediction is correct, log loss considers the confidence:

- Predicting 0.99 when correct: Low loss (good)
- Predicting 0.51 when correct: Higher loss (less confident)
- Predicting 0.99 when wrong: Very high loss (confidently wrong = bad)

## How It Works

For binary classification:
Log Loss = -1/N * Σ[y*log(p) + (1-y)*log(1-p)]

For multi-class:
Log Loss = -1/N * Σ Σ y_ik * log(p_ik)

**Interpretation:**

- Log Loss = 0: Perfect predictions with complete confidence
- Lower is better
- Log Loss approaches infinity for confidently wrong predictions

**Use cases:** Essential for evaluating probabilistic classifiers, probability calibration,
and whenever predicted probabilities (not just class labels) matter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LogLossMetric(Double)` | Initializes a new Log Loss metric. |

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
| `Compute(ReadOnlySpan<>,ReadOnlySpan<>,Int32)` |  |
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,Int32,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` |  |

