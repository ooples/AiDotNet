---
title: "RfaFullModelAggregationStrategy<T, TInput, TOutput>"
description: "Robust Federated Aggregation (RFA) via geometric median (Weiszfeld iterations)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Robust Federated Aggregation (RFA) via geometric median (Weiszfeld iterations).

## How It Works

**For Beginners:** Instead of averaging client updates, the geometric median finds a point that
minimizes the sum of distances to all client updates. This is more robust when some clients are
outliers or adversarial.

