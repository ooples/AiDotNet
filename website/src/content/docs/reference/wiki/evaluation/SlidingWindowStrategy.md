---
title: "SlidingWindowStrategy<T>"
description: "Sliding Window cross-validation for time series with fixed-size training window."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Sliding Window cross-validation for time series with fixed-size training window.

## For Beginners

Unlike expanding window (TimeSeriesSplit), sliding window:

- Uses a fixed-size training window that "slides" forward
- Better for non-stationary time series where old data becomes less relevant
- Each fold trains on the same amount of data
- Useful when you believe recent history is more predictive than distant past

## How It Works

**Example:** With window_size=100 and test_size=20:

- Fold 1: Train [0-99], Test [100-119]
- Fold 2: Train [20-119], Test [120-139]
- Fold 3: Train [40-139], Test [140-159]

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlidingWindowStrategy(Int32,Int32,Nullable<Int32>)` | Initializes Sliding Window cross-validation. |

