---
title: "TimeSeriesValidation"
description: "Provides specialized time series cross-validation strategies beyond basic TimeSeriesSplit."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Preprocessing.TimeSeries`

Provides specialized time series cross-validation strategies beyond basic TimeSeriesSplit.

## For Beginners

Different time series problems need different validation strategies:

**Blocked Time Series Split:** Like TimeSeriesSplit but with additional purging around
the test set to prevent data leakage from overlapping features.

**Walk-Forward Validation:** Simulates real-world deployment by retraining the model
at each step with all available data up to that point.

**Purged Group Time Series Split:** Groups data by time period and ensures no group
is split across train/test sets.

## Methods

| Method | Summary |
|:-----|:--------|
| `BlockedTimeSeriesSplit(Int32,Int32,Double)` | Creates a blocked time series split with purging around test sets. |
| `WalkForward(Int32,Int32,Int32,Nullable<Int32>)` | Generates walk-forward validation splits. |

