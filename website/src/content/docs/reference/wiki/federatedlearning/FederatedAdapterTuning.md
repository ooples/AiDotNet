---
title: "FederatedAdapterTuning<T>"
description: "Implements FedAdapter — federated bottleneck adapter tuning where small adapter modules are inserted into each transformer block and only these are communicated."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements FedAdapter — federated bottleneck adapter tuning where small adapter modules
are inserted into each transformer block and only these are communicated.

## For Beginners

Instead of modifying a large model's weights directly, adapters
insert small "bottleneck" layers (down-project → activation → up-project) after the
attention and feed-forward layers in each transformer block. Only these tiny bottleneck
layers are trained and shared in federated learning, keeping the base model frozen.

## How It Works

Architecture per adapted layer:

Reference: Cai, X., et al. (2023). "FedAdapter: Efficient Federated Learning via
Bottleneck Adapters." NeurIPS Workshop 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedAdapterTuning(Int32,Int32,Int32,Int32)` | Creates a new FedAdapter strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `BottleneckDimension` | Gets the bottleneck hidden dimension. |
| `CompressionRatio` |  |
| `LayerDimension` | Gets the transformer hidden dimension. |
| `NumAdaptedLayers` | Gets the number of adapted transformer layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ApplyAdapterForward([],[],Double)` | Applies the adapter forward pass with residual connection: output = x + scale * UpProject(ReLU(DownProject(x))). |
| `ExtractAdapterParameters(Vector<>)` |  |
| `GetLayerAdapterParams(Vector<>,Int32,Int32)` | Extracts adapter parameters for a specific layer and position (attention or FFN). |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |

