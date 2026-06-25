---
title: "TrimmedMeanFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Coordinate-wise trimmed mean aggregation for `IFullModel`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Coordinate-wise trimmed mean aggregation for `IFullModel`.

## How It Works

**For Beginners:** This strategy sorts each parameter across clients, drops the extreme values
on both ends, then averages the remaining values. This reduces the impact of outliers.

