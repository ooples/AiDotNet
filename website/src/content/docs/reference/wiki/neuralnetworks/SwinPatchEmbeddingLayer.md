---
title: "SwinPatchEmbeddingLayer<T>"
description: "Patch embedding layer for Swin Transformer that converts images to patch sequences."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Patch embedding layer for Swin Transformer that converts images to patch sequences.

## For Beginners

Think of this layer as cutting an image into small squares (patches)
and converting each square into a list of numbers (embedding) that describes its content.
This allows the transformer to process images as sequences, similar to how it processes text.

## How It Works

This layer divides an input image into non-overlapping patches and projects each patch
to an embedding vector. This is the first step in processing images with Swin Transformer.

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinPatchEmbeddingLayer(Int32,Int32)` | Creates a new Swin patch embedding layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumPatches` | Gets the number of patches produced by this layer. |
| `ParameterCount` |  |
| `PatchGridHeight` | Gets the height of the patch grid. |
| `PatchGridWidth` | Gets the width of the patch grid. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `Forward(Tensor<>)` | Performs the forward pass, converting image to patch sequence. |
| `GetMetadata` | Emits the constructor settings (patch size + embedding dim) that cannot be inferred from shapes alone. |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_norm` | Layer normalization applied after patch embedding. |
| `_projection` | The convolutional layer used for patch projection. |

