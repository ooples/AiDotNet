---
title: "BulyanFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Bulyan aggregation for `IFullModel` (Multi-Krum selection + trimmed aggregation)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Bulyan aggregation for `IFullModel` (Multi-Krum selection + trimmed aggregation).

## How It Works

**For Beginners:** Bulyan is a stronger robust aggregation approach:
1) Use Multi-Krum to pick a set of "reasonable" client updates.
2) For each parameter, apply a trimmed aggregation over that set.

This can better tolerate malicious clients, at the cost of more computation.

