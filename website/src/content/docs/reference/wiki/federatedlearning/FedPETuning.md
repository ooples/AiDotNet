---
title: "FedPETuning<T>"
description: "Implements FedPETuning — a unified framework for parameter-efficient fine-tuning (PEFT) in federated learning that supports multiple PEFT methods under one API."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements FedPETuning — a unified framework for parameter-efficient fine-tuning (PEFT) in
federated learning that supports multiple PEFT methods under one API.

## For Beginners

There are many ways to fine-tune a large model cheaply (LoRA,
adapter layers, prefix tuning, BitFit, etc.). FedPETuning wraps them all into a single
federated strategy so you can swap methods easily. It also applies federated-aware selection
to decide which parameters each client should update, based on data heterogeneity.

## How It Works

Supported methods:

Reference: Zhang, Z., et al. (2023). "Federated Learning for Parameter-Efficient
Fine-Tuning of Foundation Models." ACL 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedPETuning(Int32,PEFTMethod,Int32,Int32,Int32)` | Creates a new FedPETuning strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `BottleneckDim` | Gets the bottleneck/rank dimension. |
| `CompressionRatio` |  |
| `Method` | Gets the PEFT method being used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `AggregateAdaptersMethodAware(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` | Aggregates adapters with method-specific logic: LoRA matrices use SVD-aware averaging, BitFit biases use straight averaging, and adapters/prefixes use weighted averaging. |
| `ApplySelectionMask(Vector<>,Boolean[])` | Applies a selection mask to adapter parameters, zeroing out unselected positions. |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |
| `SelectParametersByHeterogeneity(Dictionary<Int32,Double[]>,Double)` | Selects which adapter parameters each client should update based on data heterogeneity. |

