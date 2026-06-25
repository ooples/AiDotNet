---
title: "PrecisionMetric<T>"
description: "Computes precision (positive predictive value): the proportion of positive predictions that are correct."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes precision (positive predictive value): the proportion of positive predictions that are correct.

## For Beginners

Precision answers: "When the model predicts positive, how often is it correct?"
High precision means few false positives (false alarms). A spam filter with 99% precision
means only 1% of emails it flags as spam are actually legitimate.

## How It Works

Precision = TP / (TP + FP) = True Positives / Predicted Positives

**When to prioritize precision:**

- When false positives are costly (blocking legitimate transactions, accusing innocent people)
- When you need high confidence in positive predictions

**Multi-class:** For multi-class problems, use averaging (micro, macro, weighted) to combine
per-class precision values.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrecisionMetric(,AveragingMethod)` | Initializes a new precision metric with an explicit positive label. |
| `PrecisionMetric(AveragingMethod)` | Initializes a new precision metric with default positive label (1). |

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

