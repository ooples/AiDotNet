---
title: "TabDPTBase<T>"
description: "Base class for TabDPT (Tabular Data Pre-Training) foundation model."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base class for TabDPT (Tabular Data Pre-Training) foundation model.

## For Beginners

TabDPT is like a "GPT for tables":

- **Pre-training**: Model learns patterns from many different tabular datasets
- **Transfer learning**: These learned patterns help on new, unseen data
- **In-context learning**: Given a few examples, it adapts to new tasks
- **Feature-wise attention**: Understands relationships between columns

The model processes features as tokens and uses transformer architecture
to capture complex interactions, similar to how language models process words.

## How It Works

TabDPT applies foundation model concepts to tabular data, using pre-training
on diverse datasets to learn transferable representations that can adapt
to new tasks through in-context learning.

Reference: "TabDPT: Scaling Tabular Foundation Models" (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabDPTBase(Int32,TabDPTOptions<>)` | Initializes a new instance of the TabDPTBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `MLPOutputDimension` | Gets the MLP output dimension. |
| `NumNumericalFeatures` | Gets the number of numerical features. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateOneHotEncoding(Matrix<Int32>,Int32,Int32)` | Creates one-hot encoding for categorical features. |
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the backbone network. |
| `ResetState` | Resets internal state and caches. |
| `UpdateParameters()` | Updates all trainable parameters. |

