---
title: "TabPFNBase<T>"
description: "Base class for TabPFN (Prior-Fitted Networks) for tabular data."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base class for TabPFN (Prior-Fitted Networks) for tabular data.

## For Beginners

TabPFN works differently from traditional models:

- **Pre-training**: Model is trained on millions of synthetic datasets
- **In-context learning**: Training data becomes part of the input
- **No gradient updates**: Inference only, no fine-tuning needed
- **Transformer backbone**: Uses attention to learn patterns from context

The key insight is that TabPFN learns to be a "learning algorithm" itself,
similar to how GPT learns to complete text.

## How It Works

TabPFN is a meta-learning approach using transformers pre-trained on synthetic
data. It performs in-context learning by conditioning on training examples
to make predictions on test samples.

Reference: "TabPFN: A Transformer That Solves Small Tabular Classification
Problems in a Second" (2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabPFNBase(Int32,TabPFNOptions<>)` | Initializes a new instance of the TabPFNBase class. |

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
| `ClearContext` | Clears the context data. |
| `CreateOneHotEncoding(Matrix<Int32>,Int32,Int32)` | Creates one-hot encoding for categorical features. |
| `CreatePositionalEncoding(Int32,Int32)` | Creates sinusoidal positional encoding. |
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the backbone network. |
| `ResetState` | Resets internal state and caches. |
| `SetContext(Tensor<>,Tensor<>)` | Sets the context (training) data for in-context learning. |
| `UpdateParameters()` | Updates all trainable parameters. |

