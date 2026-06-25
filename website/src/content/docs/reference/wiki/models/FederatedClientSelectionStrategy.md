---
title: "FederatedClientSelectionStrategy"
description: "Specifies how clients are selected to participate in a federated training round."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how clients are selected to participate in a federated training round.

## How It Works

**For Beginners:** In large deployments, not every client participates in every round.
Client selection strategies control which clients are chosen each round.

## Fields

| Field | Summary |
|:-----|:--------|
| `AvailabilityAware` | Prefer clients with higher reported availability. |
| `Clustered` | Cluster clients (e.g., by embeddings) and sample across clusters. |
| `PerformanceAware` | Prefer clients with better observed performance, with exploration. |
| `Stratified` | Stratified sampling based on group keys. |
| `UniformRandom` | Uniform random sampling. |
| `WeightedRandom` | Weighted random sampling (typically proportional to client sample counts). |

