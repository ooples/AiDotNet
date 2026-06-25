---
title: "ILayeredModel<T>"
description: "Provides layer-level access to a neural network's architecture and parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Provides layer-level access to a neural network's architecture and parameters.

## For Beginners

Neural networks are made up of layers stacked on top of each other.
Most model interfaces only let you access all parameters as one big list. This interface
lets you inspect and manipulate individual layers - their shapes, weights, types, and
connections. This enables advanced techniques like:

## How It Works

This interface exposes individual layers with their metadata, enabling per-layer operations
across the AiDotNet stack: pipeline parallelism, quantization, pruning, LoRA, meta-learning,
activation checkpointing, model export, and knowledge distillation.

**Reference:** This design is inspired by PyTorch's `nn.Module` hierarchy,
Megatron-LM's pipeline partition API, and Flax NNX's module introspection.

## Properties

| Property | Summary |
|:-----|:--------|
| `LayerCount` | Gets the number of layers in this model. |
| `Layers` | Gets the ordered list of layers in this model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSubModel(Int32,Int32)` | Extracts a contiguous sub-model from `startLayer` to `endLayer` (inclusive). |
| `GetAllLayerInfo` | Gets metadata for all layers. |
| `GetLayerInfo(Int32)` | Gets metadata for a specific layer including its parameter offset within the flat parameter vector, enabling layer-aware slicing. |
| `ValidatePartitionPoint(Int32)` | Validates that a partition point between layers is valid (output shape of layer at `afterLayerIndex` is compatible with input shape of the next layer). |

