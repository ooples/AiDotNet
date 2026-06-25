---
title: "HeterogeneousLoRA<T>"
description: "Heterogeneous LoRA — supports different LoRA ranks per client with SVD-based aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Heterogeneous LoRA — supports different LoRA ranks per client with SVD-based aggregation.

## For Beginners

Not all devices are equal — a powerful server can afford rank 64
while a phone can only handle rank 4. Heterogeneous LoRA lets each device choose its own
rank, then intelligently combines them at the server using singular value decomposition
to extract the most important adaptation directions.

## How It Works

HeLoRA (Yue et al., 2024) and FlexLoRA (Bai et al., 2024) allow each client to use a
different LoRA rank based on their computational budget. The server aggregates adapters
of varying sizes using SVD-based weight redistribution: it reconstructs full-rank deltas,
averages them, then re-decomposes into the target rank.

References:
Yue et al. (2024), "HeLoRA: Heterogeneous Low-Rank Adapters for Federated Fine-Tuning".
Bai et al. (2024), "FlexLoRA: Stacking-based Heterogeneous LoRA Aggregation" (NeurIPS 2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HeterogeneousLoRA(Int32,Int32,Int32,Int32)` | Creates a new heterogeneous LoRA strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `CompressionRatio` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |

