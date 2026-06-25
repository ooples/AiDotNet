---
title: "NODEBase<T>"
description: "Base class for NODE (Neural Oblivious Decision Ensembles)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base class for NODE (Neural Oblivious Decision Ensembles).

## For Beginners

Think of NODE as making decision trees trainable like neural networks:

- **Oblivious trees**: At each depth, all nodes split on the same feature.

This makes trees faster and more regularized.

- **Soft splits**: Instead of "if feature > threshold then go right",

we use a smooth function that gradually transitions.

- **Ensemble**: Multiple trees vote together for the final answer.

The result is a model that combines tree interpretability with deep learning power.

## How It Works

NODE uses differentiable oblivious decision trees that can be trained end-to-end:

- Each tree uses the same feature at each depth level (oblivious)
- Split decisions are soft (differentiable) using sigmoid
- Trees are combined additively for the final output

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NODEBase(Int32,NODEOptions<>)` | Initializes a new instance of the NODEBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters. |
| `TreeOutputDimension` | Gets the tree output dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardBackbone(Tensor<>)` | Performs the forward pass through the NODE backbone. |
| `GetFeatureImportance` | Gets the feature importance scores. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates all parameters. |

