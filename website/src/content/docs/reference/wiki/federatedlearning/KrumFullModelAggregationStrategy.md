---
title: "KrumFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Krum aggregation for `IFullModel` (Byzantine-robust selection by distance)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Krum aggregation for `IFullModel` (Byzantine-robust selection by distance).

## How It Works

**For Beginners:** Krum picks the single client update that is most consistent with the others.
It does this by computing distances between client updates and selecting the one with the smallest
sum of distances to its closest neighbors.

