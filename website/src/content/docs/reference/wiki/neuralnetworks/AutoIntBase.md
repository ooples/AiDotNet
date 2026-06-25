---
title: "AutoIntBase<T>"
description: "Base class for AutoInt (Automatic Feature Interaction Learning)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base class for AutoInt (Automatic Feature Interaction Learning).

## For Beginners

AutoInt discovers which features work well together:

- **Without AutoInt**: You manually create features like "age * income"
- **With AutoInt**: The model automatically learns "age and income interact"

This is especially useful for recommendation systems, click prediction,
and any tabular task where feature combinations matter.

## How It Works

AutoInt uses multi-head self-attention to automatically learn feature interactions:

1. Each feature is embedded into a dense vector
2. Self-attention layers learn interactions between features
3. Interactions are combined with original embeddings for prediction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoIntBase(Int32,AutoIntOptions<>)` | Initializes a new instance of the AutoIntBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedFeatures(Tensor<>,Matrix<Int32>)` | Embeds input features. |
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the AutoInt backbone. |
| `GetInteractionWeights` | Gets the learned feature interaction importance. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates all parameters. |

