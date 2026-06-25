---
title: "RollingOriginSplitter<T>"
description: "Rolling origin evaluation for multi-step forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Rolling origin evaluation for multi-step forecasting.

## For Beginners

Rolling origin is designed for evaluating forecasts at multiple
horizons (1-step, 2-step, etc. ahead).

## How It Works

**How It Works:**
The "origin" is the last training point, and you forecast H steps ahead.

**When to Use:**

- Multi-step forecasting evaluation
- When you need to test different forecast horizons
- Standard time series forecasting evaluation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RollingOriginSplitter` | Initializes a new instance with default settings. |
| `RollingOriginSplitter(Int32,Int32,Int32)` | Creates a new rolling origin splitter. |

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

