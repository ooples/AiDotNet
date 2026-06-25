---
title: "MultiKrumFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Multi-Krum aggregation for `IFullModel` (select m central updates, then average)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Multi-Krum aggregation for `IFullModel` (select m central updates, then average).

## How It Works

**For Beginners:** Multi-Krum is like Krum, but instead of picking only one client update,
it picks a small group of the most "central" updates and averages them.

