---
title: "RecallMetric<T>"
description: "Computes recall (sensitivity, true positive rate): the proportion of actual positives correctly identified."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes recall (sensitivity, true positive rate): the proportion of actual positives correctly identified.

## For Beginners

Recall answers: "Of all actual positives, how many did the model find?"
High recall means few false negatives (missed positives). A cancer screening test with 99% recall
means it catches 99% of actual cancer cases.

## How It Works

Recall = TP / (TP + FN) = True Positives / Actual Positives

**When to prioritize recall:**

- When false negatives are costly (missing disease, missing fraud)
- When you must catch as many positives as possible

**Trade-off:** Improving recall often decreases precision, and vice versa.
F1-score balances both, and you can use F-beta to weight one more than the other.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RecallMetric(,AveragingMethod)` | Initializes a new recall metric with an explicit positive label. |
| `RecallMetric(AveragingMethod)` | Initializes a new recall metric with default positive label (1). |

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

