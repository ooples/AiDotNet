---
title: "PiecewiseLinearEncodingLayer<T>"
description: "Piecewise Linear Encoding for numerical features in tabular models like TabM."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Piecewise Linear Encoding for numerical features in tabular models like TabM.

## For Beginners

Think of this like creating "bins" for each number:

- A feature value of 25 might activate "20-30" bin strongly
- It might partially activate neighboring bins too
- This gives the model more ways to understand numerical values

It's similar to how histograms work, but with soft (differentiable) boundaries.

## How It Works

Piecewise linear encoding transforms numerical features into a richer representation
by computing activations based on learned bin boundaries. Each feature is encoded
as a combination of linear pieces, allowing the model to learn non-linear relationships.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PiecewiseLinearEncodingLayer(Int32,Int32)` | Initializes piecewise linear encoding. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension (numFeatures * numBins). |
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Encodes numerical features using piecewise linear representation. |
| `GetParameters` |  |
| `ResetState` |  |
| `UpdateParameters()` |  |

