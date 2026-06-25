---
title: "TimeSeriesSplitStrategy<T>"
description: "Time Series Split: expanding window cross-validation that respects temporal order."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.CrossValidation`

Time Series Split: expanding window cross-validation that respects temporal order.

## For Beginners

Time series data cannot be shuffled because order matters.
Time Series Split uses an expanding training window:

- Training data always comes BEFORE validation data
- Simulates real-world forecasting where you predict future from past
- Avoids data leakage from future information

## How It Works

**Example with 5 splits:**
Notice how training data grows with each split (expanding window).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesSplitStrategy(Int32,Nullable<Int32>,Nullable<Int32>,Int32)` | Initializes Time Series Split cross-validation. |

