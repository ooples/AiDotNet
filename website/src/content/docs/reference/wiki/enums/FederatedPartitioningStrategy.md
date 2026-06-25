---
title: "FederatedPartitioningStrategy"
description: "Defines how a centralized dataset should be partitioned into per-client datasets for federated simulations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines how a centralized dataset should be partitioned into per-client datasets for federated simulations.

## Fields

| Field | Summary |
|:-----|:--------|
| `DirichletLabel` | Dirichlet label distribution partitioning (common FL benchmark approach). |
| `IID` | IID partitioning (uniform random assignment of samples to clients). |
| `ShardByLabel` | Shard-by-label partitioning (sort by label then assign label shards to clients). |

