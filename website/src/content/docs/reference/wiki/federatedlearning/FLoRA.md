---
title: "FLoRA<T>"
description: "Implements FLoRA — Federated Low-Rank Adaptation with stacked lossless aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements FLoRA — Federated Low-Rank Adaptation with stacked lossless aggregation.

## For Beginners

Standard federated LoRA averages the A and B matrices from
different clients, which introduces approximation error. FLoRA instead *stacks*
the local LoRA updates — each client's (B_k, A_k) pair is concatenated vertically/horizontally,
preserving all information. The server then uses an SVD-based compression to bring the stacked
result back to the target rank. This gives lossless aggregation without the information loss
of simple averaging.

## How It Works

Algorithm:

Reference: Wang, Y., et al. (2024). "FLoRA: Federated Fine-Tuning Large Language
Models with Heterogeneous Low-Rank Adaptations." arXiv:2405.14739.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FLoRA(Int32,Int32,Double,Int32,Int32,Int32)` | Creates a new FLoRA strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `Alpha` | Gets the LoRA alpha scaling factor. |
| `CompressionRatio` |  |
| `Rank` | Gets the target LoRA rank. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |
| `TruncatedSVD(Double[],Int32,Int32,Int32)` | Truncated SVD via power iteration. |

