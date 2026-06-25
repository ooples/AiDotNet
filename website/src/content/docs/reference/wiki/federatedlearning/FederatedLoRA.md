---
title: "FederatedLoRA<T>"
description: "Federated LoRA — Low-Rank Adaptation for parameter-efficient federated fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Federated LoRA — Low-Rank Adaptation for parameter-efficient federated fine-tuning.

## For Beginners

Imagine a large model as a huge spreadsheet. Instead of modifying
every cell (billions of values), LoRA only learns a small "correction" matrix that captures
the most important changes. In federated LoRA, each device only sends this tiny correction
instead of the whole spreadsheet — making it practical to collaboratively fine-tune
GPT-scale models across phones or hospitals.

## How It Works

LoRA (Hu et al., 2021) decomposes weight updates into low-rank matrices: ΔW = BA where
B ∈ R^{d×r} and A ∈ R^{r×k} with rank r ≪ min(d,k). In federated settings, only the
LoRA matrices are communicated, reducing bandwidth by 100-1000x for large models.

This implementation follows the FedEx-LoRA approach (ACL 2025) which performs exact
aggregation by averaging A and B matrices separately with residual error correction.

References:
Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models".
Sun et al. (2025), "FedEx-LoRA: Exact Aggregation for Federated Low-Rank Adaptation" (ACL 2025).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedLoRA(Int32,Int32,Double,Int32,Int32,Int32)` | Creates a new federated LoRA strategy. |

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

