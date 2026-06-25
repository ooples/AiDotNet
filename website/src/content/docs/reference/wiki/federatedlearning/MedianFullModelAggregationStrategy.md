---
title: "MedianFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Coordinate-wise median aggregation for `IFullModel`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Coordinate-wise median aggregation for `IFullModel`.

## How It Works

**For Beginners:** For each model parameter, this strategy takes the middle value across clients.
This makes the aggregation resistant to outliers (e.g., a client sending extremely large values).

