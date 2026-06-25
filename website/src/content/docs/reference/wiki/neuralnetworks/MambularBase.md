---
title: "MambularBase<T>"
description: "Base class for Mambular (State Space Models for Tabular Data)."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base class for Mambular (State Space Models for Tabular Data).

## For Beginners

Mambular is an alternative to transformers that:

- **Scales linearly**: O(n) instead of O(n²) with sequence length
- **Has memory**: Can remember information across the feature sequence
- **Is selective**: Learns what to remember and what to forget

This makes it efficient for tabular data with many features.

## How It Works

Mambular applies the Mamba architecture to tabular data:

1. Features are embedded and treated as a sequence
2. Selective State Space Model (S4/Mamba) processes the sequence
3. Final representation is used for prediction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MambularBase(Int32,MambularOptions<>)` | Initializes a new instance of the MambularBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedFeatures(Tensor<>,Matrix<Int32>)` | Embeds input features. |
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the Mambular backbone. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates all parameters. |

