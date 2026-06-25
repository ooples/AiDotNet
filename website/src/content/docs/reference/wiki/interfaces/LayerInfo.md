---
title: "LayerInfo<T>"
description: "Metadata about a single layer within a layered model, including its position in the flat parameter vector and computational characteristics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Metadata about a single layer within a layered model, including its position
in the flat parameter vector and computational characteristics.

## How It Works

**For Beginners:** When you have a neural network with many layers, you often need
to know details about each one: how many parameters it has, what type it is, where its
parameters sit in the overall parameter vector, and how expensive it is to compute.

This class packages all that information together so that tools like pipeline partitioners,
quantizers, and pruners can make smart per-layer decisions.

For example, a pipeline partitioner can use `EstimatedFlops` to balance
computational load across GPUs, or a quantizer can use `Category` to apply
different bit-widths to attention vs dense layers.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Layer type classification for automated decisions. |
| `EstimatedActivationMemory` | Estimated activation memory (bytes) needed during forward pass. |
| `EstimatedFlops` | Estimated computational cost (FLOPs) for a single forward pass. |
| `Index` | Layer index within the model (0-based). |
| `InputShape` | Input shape expected by this layer. |
| `IsTrainable` | Whether this layer has trainable parameters. |
| `Layer` | Reference to the actual layer instance. |
| `Name` | Human-readable layer name (e.g., "SelfAttentionLayer_0", "FullyConnectedLayer_3"). |
| `OutputShape` | Output shape produced by this layer. |
| `ParameterCount` | Number of trainable parameters in this layer. |
| `ParameterOffset` | Start index of this layer's parameters in the flat parameter vector returned by `GetParameters()`. |

