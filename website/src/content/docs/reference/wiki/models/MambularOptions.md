---
title: "MambularOptions<T>"
description: "Configuration options for Mambular (State Space Models for Tabular Data)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Mambular (State Space Models for Tabular Data).

## For Beginners

Mambular treats your features like a sequence:

- **State Space Models**: An efficient alternative to transformers
- **Linear complexity**: Scales better than attention with many features
- **Selective mechanism**: Learns which features to remember/forget

Example:

## How It Works

Mambular applies the Mamba (State Space Model) architecture to tabular data,
treating features as a sequence and using selective state spaces for processing.

Reference: "Mambular: A Sequential Model for Tabular Deep Learning" (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoricalCardinalities` | Gets or sets the cardinalities of categorical features. |
| `ConvKernelSize` | Gets or sets the convolution kernel size. |
| `DeltaMax` | Gets or sets the delta (discretization step) range maximum. |
| `DeltaMin` | Gets or sets the delta (discretization step) range minimum. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for features. |
| `ExpansionFactor` | Gets or sets the expansion factor for the inner dimension. |
| `HiddenActivation` | Gets or sets the hidden layer activation function for the MLP head. |
| `HiddenVectorActivation` | Gets or sets the hidden layer vector activation function (alternative to scalar activation). |
| `InitScale` | Gets or sets the initialization scale for parameters. |
| `InnerDimension` | Gets the inner dimension. |
| `MLPHiddenDimensions` | Gets or sets the hidden dimensions for the MLP head. |
| `NumLayers` | Gets or sets the number of Mamba layers. |
| `StateDimension` | Gets or sets the state dimension for the SSM. |
| `UseBidirectional` | Gets or sets whether to use bidirectional processing. |

