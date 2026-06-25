---
title: "SwinTransformerBlockLayer<T>"
description: "Swin Transformer block layer with windowed multi-head self-attention."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Swin Transformer block layer with windowed multi-head self-attention.

## For Beginners

Unlike standard transformers that compute attention across all tokens
(which is expensive for images), Swin Transformer divides the image into windows and
computes attention only within each window. To allow information flow between windows,
alternate layers use "shifted" windows that overlap the original window boundaries.

## How It Works

This layer implements the core Swin Transformer block with:

- Window-based multi-head self-attention (W-MSA or SW-MSA)
- Shifted window partitioning for cross-window connections
- Two-layer MLP with GELU activation
- Pre-norm architecture with residual connections
- Learnable relative position bias

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinTransformerBlockLayer(Int32,Int32,Int32,Int32,Int32)` | Creates a new Swin Transformer block layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |
| `UsesShiftedWindows` | Gets whether this block uses shifted windows. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` | Performs the forward pass through the Swin Transformer block. |
| `GetMetadata` | Emits the constructor-level settings that cannot be inferred from the layer's input/output shapes alone, so the deserializer can rebuild this block with the exact same configuration (and therefore the exact same per-sublayer parameter count… |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

