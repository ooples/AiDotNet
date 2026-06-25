---
title: "WinsorizedMeanFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Coordinate-wise winsorized mean aggregation for `IFullModel`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Coordinate-wise winsorized mean aggregation for `IFullModel`.

## How It Works

**For Beginners:** Winsorized mean is like trimmed mean, but instead of *dropping* extreme values,
it *clips* them to the nearest remaining value before averaging. This reduces the impact of outliers
while keeping the same number of values in the average.

