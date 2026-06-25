---
title: "GraphPartitionStrategy"
description: "Specifies how a graph is partitioned across federated clients."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how a graph is partitioned across federated clients.

## For Beginners

In graph FL, a large graph must be split across clients. The partition
strategy determines how nodes are assigned:

## Fields

| Field | Summary |
|:-----|:--------|
| `CommunityBased` | Community detection-based partitioning. |
| `Metis` | METIS-based minimum edge-cut partitioning. |
| `Preassigned` | Nodes are pre-assigned to clients. |
| `Random` | Random node assignment. |
| `StreamPartition` | Stream-based partitioning for dynamic graphs. |

