---
title: "VerticalDataPartitioner<T>"
description: "Partitions features (columns) across parties for vertical federated learning simulation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Partitions features (columns) across parties for vertical federated learning simulation.

## For Beginners

In a real VFL deployment, each party already holds its own
features (the bank has financial data, the hospital has medical data). But for testing
and simulation, we need to take a single dataset and split its columns across simulated
parties. This class does that splitting.

## How It Works

**Partitioning strategies:**

**Reference:** VertiBench (ICLR 2024) recommends diverse feature distribution testing.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractPartyFeatures(Tensor<>,IReadOnlyList<Int32>)` | Extracts a party's feature columns from a full data tensor. |
| `PartitionCustom(IDictionary<Int32,IReadOnlyList<Int32>>)` | Creates a vertical partition from explicit column assignments. |
| `PartitionInterleaved(Int32,Int32)` | Partitions features in an interleaved (round-robin) pattern across parties. |
| `PartitionRandom(Int32,Int32,Nullable<Int32>)` | Partitions features randomly across parties. |
| `PartitionSequential(Int32,Int32)` | Partitions features sequentially across the specified number of parties. |

