---
title: "BalancedAccuracyMetric<T>"
description: "Computes balanced accuracy: the average recall across all classes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes balanced accuracy: the average recall across all classes.

## For Beginners

Balanced accuracy gives equal weight to each class, regardless of size.
This makes it much better than regular accuracy for imbalanced datasets. A model that only
predicts the majority class will have ~50% balanced accuracy for binary classification,
not the misleadingly high regular accuracy.

## How It Works

Balanced Accuracy = (Sum of per-class recall) / (Number of classes)
For binary: Balanced Accuracy = (Sensitivity + Specificity) / 2

**Example:** With 95 negative and 5 positive samples:

- A model predicting all negative: Accuracy = 95%, Balanced Accuracy = 50%
- A model with 90% sensitivity and 90% specificity: Balanced Accuracy = 90%

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

