---
title: "WalkForwardSplitter<T>"
description: "Walk-forward validation that simulates production deployment over time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Walk-forward validation that simulates production deployment over time.

## For Beginners

Walk-forward validation simulates how your model would actually
be deployed and retrained over time.

## How It Works

**How It Works:**

**Why Use Walk-Forward?**

- Most realistic evaluation for production deployment
- Tests how the model adapts as new data arrives
- Standard for algorithmic trading and forecasting

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WalkForwardSplitter` | Initializes a new instance with default settings. |
| `WalkForwardSplitter(Int32,Int32,Nullable<Int32>)` | Creates a new walk-forward validation splitter. |

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

