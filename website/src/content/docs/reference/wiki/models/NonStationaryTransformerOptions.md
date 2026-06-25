---
title: "NonStationaryTransformerOptions<T>"
description: "Configuration options for the Non-stationary Transformer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Non-stationary Transformer model.

## For Beginners

These options control how the Non-stationary Transformer model behaves:

Key settings:

- **NumEncoderLayers:** How many encoding layers to use for pattern extraction
- **NumDecoderLayers:** How many decoding layers for generating forecasts
- **NumHeads:** Number of attention heads for multi-head attention
- **UseDeStat:** Whether to use De-stationary Attention mechanism

Time series data often has changing statistical properties (non-stationarity). This model
explicitly handles this by:

1. Normalizing the data to be stationary for better attention
2. De-normalizing attention outputs to preserve original data characteristics

## How It Works

Non-stationary Transformer addresses the over-stationarization problem in time series
forecasting by proposing Series Stationarization and De-stationary Attention mechanisms.

**Reference:** Liu et al., "Non-stationary Transformers: Exploring the Stationarity
in Time Series Forecasting", NeurIPS 2022. https://arxiv.org/abs/2205.14415

## Properties

| Property | Summary |
|:-----|:--------|
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward network dimension. |
| `LabelLength` | Gets or sets the label length (decoder input overlap). |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension (embedding size). |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `ProjectionDimension` | Gets or sets the number of projection dimensions for stationarization. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SequenceLength` | Gets or sets the input sequence length. |
| `UseDeStationaryAttention` | Gets or sets whether to use De-stationary Attention. |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |
| `UseSeriesStationarization` | Gets or sets whether to use Series Stationarization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and returns any validation errors. |

