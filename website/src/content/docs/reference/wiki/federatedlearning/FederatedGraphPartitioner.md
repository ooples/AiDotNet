---
title: "FederatedGraphPartitioner<T>"
description: "Partitions a graph across federated clients using various strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Graph`

Partitions a graph across federated clients using various strategies.

## For Beginners

Before federated graph learning can begin, the graph must be divided
among clients. The partitioner splits nodes into groups, trying to minimize the number of edges
that cross partition boundaries (cross-client edges) while keeping partitions balanced.

## How It Works

**Strategies:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedGraphPartitioner(FederatedGraphOptions)` | Initializes a new instance of `FederatedGraphPartitioner`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSubgraph(Tensor<>,Tensor<>,Int32[],Int32)` | Extracts subgraph data for a specific partition. |
| `Partition(Tensor<>,Tensor<>,Int32)` | Partitions a graph into the specified number of parts. |

