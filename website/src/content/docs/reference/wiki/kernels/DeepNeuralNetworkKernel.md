---
title: "DeepNeuralNetworkKernel<T>"
description: "Implements a deep (multi-layer) Neural Network kernel."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements a deep (multi-layer) Neural Network kernel.

## For Beginners

This kernel corresponds to a deep neural network with multiple layers.

In each layer:

1. Input from previous layer goes through the arc-cosine kernel transformation
2. Output becomes input to the next layer

More layers = more compositional expressiveness, similar to deep neural networks.
However, very deep kernels can have vanishing/exploding gradients just like deep NNs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepNeuralNetworkKernel(Int32,Double,Double,Int32)` | Initializes a deep Neural Network kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLayers` | Gets the number of layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the deep Neural Network kernel value between two vectors. |
| `ComputeJ(Double,Double)` | Computes the arc-cosine function J_n(θ). |

