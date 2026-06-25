---
title: "SAINTBase<T>"
description: "Base class for SAINT (Self-Attention and Intersample Attention Transformer)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base class for SAINT (Self-Attention and Intersample Attention Transformer).

## For Beginners

Think of SAINT as looking at your data from two perspectives:

- **Column attention**: "Which features are related to each other?"

(e.g., income and education level might be correlated)

- **Row attention**: "Which samples in my batch are similar?"

(e.g., customers with similar profiles might have similar behavior)

By combining both views, SAINT can learn patterns that other models miss.

## How It Works

SAINT applies two types of attention in alternating layers:

1. Column attention: Self-attention over features (like FT-Transformer)
2. Row attention: Inter-sample attention comparing samples in a batch

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAINTBase(Int32,SAINTOptions<>)` | Initializes a new instance of the SAINTBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedFeatures(Tensor<>,Matrix<Int32>)` | Embeds input features into the transformer space. |
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the SAINT backbone. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates all parameters. |

