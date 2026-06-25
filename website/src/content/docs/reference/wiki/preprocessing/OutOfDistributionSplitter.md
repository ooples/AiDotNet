---
title: "OutOfDistributionSplitter<T>"
description: "Out-of-Distribution (OOD) splitter that places outlier samples in the test set."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized`

Out-of-Distribution (OOD) splitter that places outlier samples in the test set.

## For Beginners

This splitter identifies samples that are "unusual" or
"outliers" compared to the bulk of the data, and places them in the test set.
This helps evaluate how well your model handles edge cases.

## How It Works

**How It Works:**

1. Compute the centroid (average) of all samples
2. Calculate each sample's distance from the centroid
3. Place the most distant samples (outliers) in the test set

**When to Use:**

- Testing model robustness to unusual inputs
- Evaluating edge case handling
- Safety-critical applications where OOD detection matters

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OutOfDistributionSplitter(Double,Double,Int32)` | Creates a new Out-of-Distribution splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

