---
title: "SubModel<T>"
description: "Represents a contiguous sub-model extracted from a larger `ILayeredModel`."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents a contiguous sub-model extracted from a larger `ILayeredModel`.
Contains a slice of layers that can be independently forwarded through and further partitioned.

## For Beginners

When you split a neural network across GPUs for pipeline
parallelism, each GPU gets a sub-model - a consecutive sequence of layers from the
original network. This class represents that slice.

## How It Works

A sub-model implements `ILayeredModel` itself, so you can extract
sub-models of sub-models, enabling hierarchical partitioning (e.g., virtual pipeline stages
in Megatron-LM).

**Reference:** Inspired by PyTorch's `split_module()` which returns
`nn.Sequential` modules for each pipeline stage.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubModel(IReadOnlyList<ILayer<>>,IReadOnlyList<LayerInfo<>>,Int32,Int32)` | Creates a new sub-model from the specified layers and metadata. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EndIndex` | Gets the end index (inclusive) of this sub-model within the parent model. |
| `LayerCount` | Gets the number of layers in this sub-model. |
| `LayerInfos` | Gets the layer metadata for all layers in this sub-model. |
| `Layers` | Gets the ordered list of layers in this sub-model. |
| `ParameterCount` | Gets the total parameter count across all layers in this sub-model. |
| `StartIndex` | Gets the start index of this sub-model within the parent model. |
| `TotalEstimatedFlops` | Gets the total estimated FLOPs across all layers in this sub-model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSubModel(Int32,Int32)` |  |
| `Forward(Tensor<>)` | Performs a sequential forward pass through all layers in this sub-model. |
| `GetAllLayerInfo` |  |
| `GetInputShape` | Gets the input shape expected by the first layer in this sub-model. |
| `GetLayerInfo(Int32)` |  |
| `GetOutputShape` | Gets the output shape produced by the last layer in this sub-model. |
| `ValidatePartitionPoint(Int32)` |  |

