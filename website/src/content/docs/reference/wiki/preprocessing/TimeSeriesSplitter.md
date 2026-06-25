---
title: "TimeSeriesSplitter<T>"
description: "Time series splitter with expanding training window (no shuffling)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Time series splitter with expanding training window (no shuffling).

## For Beginners

Time series data is ordered by time. Unlike regular data,
you CANNOT shuffle it because the order matters - the past predicts the future,
not the other way around.

## How It Works

**How It Works (Expanding Window):**
Notice: Training window grows, always starting from the beginning.

**Critical Rule:** Test data MUST always be after training data in time.
Never let future data leak into training!

**When to Use:**

- Stock prices, weather forecasting
- Any data where time order matters
- When you want to simulate real deployment

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesSplitter(Int32,Nullable<Int32>,Nullable<Int32>,Int32)` | Creates a new time series splitter with expanding window. |

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

