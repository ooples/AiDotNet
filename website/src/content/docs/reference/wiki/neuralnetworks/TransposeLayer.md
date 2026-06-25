---
title: "TransposeLayer<T>"
description: "Reorders the axes of the input tensor according to a fixed permutation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Reorders the axes of the input tensor according to a fixed permutation. Zero-parameter
utility layer, primarily used to expose a different axis as the "last" dimension so a
`DenseLayer` can operate on it (enables MLP-Mixer-style cross-axis MLPs
without bespoke kernels).

## How It Works

The `permutation` passed to the constructor uses logical axis indices
(excluding the batch axis). For rank-N inputs with a batch axis at position 0, this layer
keeps the batch axis at index 0 and permutes the remaining N-1 axes per
`permutation`.

Common pattern (MLP-Mixer temporal mixer):

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransposeLayer(Int32[])` | Initializes a new `TransposeLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Emits the permutation alongside the base metadata so deserialization can reconstruct the layer exactly. |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` | Resolves logical input shape on first forward and computes the output shape via the permutation. |
| `ResetState` |  |
| `UpdateParameters()` |  |

