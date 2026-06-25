---
title: "FedKDCompressor<T>"
description: "Implements FedKD — Knowledge Distillation-based communication for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Compression`

Implements FedKD — Knowledge Distillation-based communication for federated learning.

## For Beginners

Instead of sending model parameters (which can be huge for LLMs),
clients send soft predictions (logits) on a shared public dataset. The server trains a global
model by distilling knowledge from these aggregated soft labels. This enables FL even when
clients have different model architectures (heterogeneous FL), since predictions are
architecture-agnostic.

## How It Works

Algorithm:

Reference: Wu, C., et al. (2022). "Communication-Efficient Federated Learning via
Knowledge Distillation." NeurIPS 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedKDCompressor(Double,Double)` | Creates a new FedKD compressor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KDWeight` | Gets the KD loss weight. |
| `Temperature` | Gets the KD temperature. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateLogits(Dictionary<Int32,[][]>,Dictionary<Int32,Double>)` | Aggregates soft predictions from multiple clients. |
| `ComputeKDLoss([],[])` | Computes KD loss between student logits and teacher soft labels. |
| `NormalizeHeterogeneousLogits(Dictionary<Int32,[][]>,Int32)` | Handles heterogeneous client architectures by padding/truncating logits to a common dimension. |
| `ServerDistillationStep([],[][],Func<[],Int32,[]>,Double,Int32)` | Performs one server-side distillation step: updates the student model parameters to match the aggregated ensemble soft labels via gradient descent on the KD loss. |

