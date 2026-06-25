---
title: "BlockedTimeSeriesSplitter<T>"
description: "Time series splitter with a gap (purge) between training and test sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Time series splitter with a gap (purge) between training and test sets.

## For Beginners

In time series, data points close together are often correlated.
If your training data is right next to your test data, some information might "leak"
from the test period into training.

## How It Works

**The Gap Solution:**
Adding a gap (also called "purge") between train and test ensures no leakage:

The gap samples are neither trained on nor tested - they're a buffer zone.

**When to Use:**

- When features depend on recent past values (rolling averages, momentum)
- Financial time series with overlapping windows
- When data points are auto-correlated

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BlockedTimeSeriesSplitter(Int32,Int32)` | Creates a new blocked time series splitter. |

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

