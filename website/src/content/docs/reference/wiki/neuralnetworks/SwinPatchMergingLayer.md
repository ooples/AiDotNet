---
title: "SwinPatchMergingLayer<T>"
description: "Patch merging layer for Swin Transformer that performs downsampling between stages."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Patch merging layer for Swin Transformer that performs downsampling between stages.

## For Beginners

Think of this like pooling in CNNs, but instead of taking
max or average, we concatenate 4 neighboring patches together (2x2 grid) and then
use a linear layer to reduce the combined channels. This lets the network process
information at multiple scales.

## How It Works

This layer merges 2x2 neighboring patches into a single patch, reducing spatial
resolution by half while doubling the channel dimension. This creates the hierarchical
structure characteristic of Swin Transformer.

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinPatchMergingLayer(Int32)` | Creates a new Swin patch merging layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `Forward(Tensor<>)` | Performs the forward pass, merging 2x2 patches. |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_norm` | Layer normalization applied before reduction. |
| `_reduction` | Linear reduction layer that projects concatenated patches to output dimension. |

