---
title: "DiffusionCrossAttention<T>"
description: "Cross-attention layer for diffusion models with Flash Attention optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Attention`

Cross-attention layer for diffusion models with Flash Attention optimization.

## For Beginners

Cross-attention is how the model "looks at" text while generating images.

- Query (Q): Comes from the image features
- Key (K) and Value (V): Come from text embeddings
- Output: Image features enriched with text information

This enables the model to generate images that match the text description.

## How It Works

Cross-attention allows the model to attend to conditioning information (like text embeddings)
when generating images. This is how text-to-image models like Stable Diffusion work.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionCrossAttention(Int32,Int32,Int32,Int32)` | Initializes a new diffusion cross-attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextDim` | Gets the context dimension. |
| `ParameterCount` |  |
| `QueryDim` | Gets the query dimension. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through cross-attention. |
| `ForwardWithContext(Tensor<>,Tensor<>)` | Performs the forward pass with context (conditioning). |
| `GetParameters` | Gets all layer parameters. |
| `ResetState` | Resets the layer's internal state. |
| `SetParameters(Vector<>)` | Sets all layer parameters. |
| `SetTrainingMode(Boolean)` | Propagates eval/training mode to the nested cross-attention sublayer so inference uses the GPU-resident fast path rather than the allocating tape/training branch. |
| `UpdateParameters()` | Updates parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_contextDim` | Context dimension (text embedding dimension). |
| `_crossAttention` | Cross-attention layer. |
| `_lastContext` | Cached context for backward pass. |
| `_lastInput` | Cached input for backward pass. |
| `_numHeads` | Number of attention heads. |
| `_queryDim` | Query dimension (spatial channels). |
| `_spatialSize` | Spatial size (height/width). |

