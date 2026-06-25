---
title: "F1ScoreMetric<T>"
description: "Computes F1 score: the harmonic mean of precision and recall."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes F1 score: the harmonic mean of precision and recall.

## For Beginners

F1 score balances precision and recall into a single number.
It's particularly useful when:

- You care about both false positives and false negatives
- Classes are imbalanced (where accuracy would be misleading)
- You need a single number to compare models

## How It Works

F1 = 2 * (Precision * Recall) / (Precision + Recall)

**Interpretation:**

- F1 = 1.0: Perfect precision and recall
- F1 = 0.5: Mediocre balance
- F1 near 0: Poor performance on at least one of precision or recall

**Note:** F1 is the harmonic mean because it penalizes extreme differences.
If precision = 0.95 and recall = 0.1, F1 = 0.18 (not 0.525 arithmetic mean).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `F1ScoreMetric(,AveragingMethod)` | Initializes a new F1 score metric with an explicit positive label. |
| `F1ScoreMetric(AveragingMethod)` | Initializes a new F1 score metric with default positive label (1). |

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
| `ComputeWithCI(ReadOnlySpan<>,ReadOnlySpan<>,ConfidenceIntervalMethod,Double,Int32,Nullable<Int32>)` |  |

