---
title: "SparseLoRA<T>"
description: "Implements SLoRA — Sparse LoRA for communication-efficient federated fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements SLoRA — Sparse LoRA for communication-efficient federated fine-tuning.

## For Beginners

Even with LoRA, sending all adapter parameters can be expensive
when many layers are adapted. SLoRA adds a sparsity step: after local training, each client
identifies which adapter elements changed most and only sends those (top-k by magnitude).
The server aggregates the sparse updates and broadcasts the result. This can reduce
communication by another 2-10x on top of LoRA's compression.

## How It Works

Algorithm:

Reference: Sparse LoRA for Federated Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SparseLoRA(Int32,Int32,Double,Double,Int32,Int32,Int32)` | Creates a new SLoRA strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `CompressionRatio` |  |
| `Rank` | Gets the LoRA rank. |
| `SparsityRatio` | Gets the sparsity ratio (fraction of elements communicated). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `AggregateSparseUpdates(Dictionary<Int32,SparseUpdate<>>,Dictionary<Int32,Double>)` | Aggregates multiple sparse updates from clients. |
| `ApplySparseUpdate(Vector<>,SparseUpdate<>)` | Applies a sparse update to existing adapter parameters: new = old + sparse_delta. |
| `ComputeCommunicationCost(SparseUpdate<>)` | Computes the actual communication cost of a sparse update (number of non-zero elements). |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |
| `SparsifyDelta(Vector<>,Vector<>)` | Applies top-k sparsification to an adapter update delta, keeping only the largest elements by magnitude. |
| `ToDense(SparseUpdate<>)` | Converts a sparse update back to a dense vector (filling non-sparse positions with zero). |

