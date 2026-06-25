---
title: "FactorTransformerOptions<T>"
description: "Configuration options for the FactorTransformer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the FactorTransformer model.

## For Beginners

Think of this like tuning a transformer for markets:

- More heads let the model look at multiple relationships at once
- More layers allow deeper reasoning but cost more compute

## How It Works

FactorTransformer uses attention to learn cross-sectional and temporal relationships.
These options define the transformer depth, head count, and factor dimensions.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate used for regularization. |
| `HiddenDimension` | Gets or sets the transformer hidden dimension. |
| `NumAssets` | Gets or sets the number of assets covered by the model. |
| `NumFactors` | Gets or sets the number of latent factors to learn. |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumTransformerLayers` | Gets or sets the number of transformer encoder layers. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `SequenceLength` | Gets or sets the input sequence length. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if any value is invalid. |

