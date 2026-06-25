---
title: "RealizedVolatilityTransformerOptions<T>"
description: "Configuration options for the Realized Volatility Transformer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Realized Volatility Transformer model.

## For Beginners

These settings control how the transformer looks back in time
and how large the attention layers are when predicting volatility.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (number of future steps). |
| `HiddenSize` | Gets or sets the transformer hidden dimension. |
| `LookbackWindow` | Gets or sets the lookback window (number of past time steps). |
| `NumAssets` | Gets or sets the number of assets modeled at once. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the option values. |

