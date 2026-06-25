---
title: "AlphaFactorOptions<T>"
description: "Configuration options for the AlphaFactorModel."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the AlphaFactorModel.

## For Beginners

Think of these settings as the knobs that shape the model:

- How many hidden factors to learn
- How many assets and features are in your data
- How large the internal layers should be

## How It Works

AlphaFactorModel learns latent factors that explain and predict excess returns.
These options let you control the factor count, input dimensionality, and hidden size.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate used for regularization. |
| `HiddenDimension` | Gets or sets the width of hidden layers. |
| `NumAssets` | Gets or sets the number of assets covered by the model. |
| `NumFactors` | Gets or sets the number of latent factors to learn. |
| `NumFeatures` | Gets or sets the number of input features per asset. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `SequenceLength` | Gets or sets the input sequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if any value is invalid. |

