---
title: "AnchoredWalkForwardSplitter<T>"
description: "Anchored walk-forward validation with a fixed starting point."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Anchored walk-forward validation with a fixed starting point.

## For Beginners

This is similar to regular walk-forward, but the training
always starts from the same point (anchored) rather than sliding.

## How It Works

**How It Works:**
Training always starts at index 0 (the anchor).

**When to Use:**

- When historical data remains relevant
- When you want maximum training data
- Traditional time series forecasting

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnchoredWalkForwardSplitter` | Initializes a new instance with default settings. |
| `AnchoredWalkForwardSplitter(Int32,Int32,Nullable<Int32>)` | Creates a new anchored walk-forward splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

