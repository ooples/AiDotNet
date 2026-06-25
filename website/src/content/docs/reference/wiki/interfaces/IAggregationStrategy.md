---
title: "IAggregationStrategy<TModel>"
description: "Defines strategies for aggregating model updates from multiple clients in federated learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines strategies for aggregating model updates from multiple clients in federated learning.

## How It Works

This interface represents different methods for combining model updates from distributed clients
into a single improved global model.

**For Beginners:** An aggregation strategy is like a voting system or consensus mechanism
that decides how to combine different opinions into a single decision.

Think of aggregation strategies as different ways to combine contributions:

- Simple average: Everyone's input counts equally
- Weighted average: Some contributors' inputs count more based on criteria (data size, accuracy)
- Robust methods: Ignore outliers or malicious contributions

For example, in a federated learning scenario with hospitals:

- Hospital A has 10,000 patients: gets weight of 10,000
- Hospital B has 5,000 patients: gets weight of 5,000
- The aggregation strategy might weight Hospital A's updates more heavily

Different strategies handle different challenges:

- FedAvg: Standard weighted averaging
- FedProx: Handles clients with different update frequencies
- Krum: Robust to Byzantine (malicious) clients
- Median aggregation: Resistant to outliers

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,>,Dictionary<Int32,Double>)` | Aggregates model updates from multiple clients into a single global model update. |
| `GetStrategyName` | Gets the name of the aggregation strategy. |

