---
title: "DistributionPreservingSplitter<T>"
description: "Stratified split for regression targets that preserves the target distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified`

Stratified split for regression targets that preserves the target distribution.

## For Beginners

Stratification is usually for classification (preserving class counts).
But what about regression where targets are continuous?

## How It Works

**Solution:** We bin the continuous target into groups, then stratify by those bins.
This ensures both train and test have similar target distributions.

**Example:**
If house prices range from $100k-$1M, we might create 10 bins:
$100k-$190k, $190k-$280k, etc.
Then ensure each split has proportional samples from each bin.

**When to Use:**

- Regression problems with skewed target distributions
- When you want representative samples across the target range
- Prevents scenarios where all expensive houses end up in test

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistributionPreservingSplitter(Double,Int32,Boolean,Int32)` | Creates a new distribution-preserving splitter for regression. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

