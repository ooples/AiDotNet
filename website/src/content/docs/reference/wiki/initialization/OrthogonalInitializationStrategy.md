---
title: "OrthogonalInitializationStrategy<T>"
description: "Orthogonal initialization strategy for RNNs, LSTMs, and deep networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

Orthogonal initialization strategy for RNNs, LSTMs, and deep networks.

## For Beginners

Use this for RNNs, LSTMs, and very deep networks.
It helps gradients flow smoothly through many layers without vanishing or exploding.

## How It Works

Orthogonal initialization creates a random orthogonal matrix, which preserves
gradient norms across layers. This prevents vanishing/exploding gradients in
deep networks and recurrent architectures.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OrthogonalInitializationStrategy(Double)` | Creates an orthogonal initialization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLazy` |  |
| `LoadFromExternal` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeBiases(Tensor<>)` |  |
| `InitializeWeights(Tensor<>,Int32,Int32)` |  |

