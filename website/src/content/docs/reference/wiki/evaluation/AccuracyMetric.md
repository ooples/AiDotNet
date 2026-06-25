---
title: "AccuracyMetric<T>"
description: "Computes classification accuracy: the proportion of correct predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes classification accuracy: the proportion of correct predictions.

## For Beginners

Accuracy is the simplest classification metric - what percentage
of predictions were correct? An accuracy of 0.9 means 90% of predictions were right.

## How It Works

Accuracy = (TP + TN) / (TP + TN + FP + FN) = Correct / Total

**Limitations:** Accuracy can be misleading for imbalanced datasets. If 95% of samples
are class A, a model that always predicts A achieves 95% accuracy but is useless.
Use balanced accuracy, F1-score, or other metrics for imbalanced data.

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

