---
title: "ETSformerOptions<T>"
description: "Configuration options for the ETSformer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the ETSformer model.

## For Beginners

These options control how the ETSformer model behaves:

Key settings:

- **NumEncoderLayers:** How many encoding layers to use for pattern extraction
- **NumDecoderLayers:** How many decoding layers for generating forecasts
- **K:** Top-K frequencies for seasonal pattern detection
- **LevelSmoothing:** Controls how quickly the model adapts to level changes

ETSformer is particularly interpretable because it explicitly models trend, seasonality,
and growth components that you can inspect and understand.

## How It Works

ETSformer (Exponential Smoothing Transformer) combines classical exponential smoothing
methods with transformer attention mechanisms for interpretable time series forecasting.

**Reference:** Woo et al., "ETSformer: Exponential Smoothing Transformers for
Time-series Forecasting", 2022. https://arxiv.org/abs/2202.01381

## Properties

| Property | Summary |
|:-----|:--------|
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward network dimension. |
| `K` | Gets or sets the top-K frequencies for seasonal decomposition. |
| `LearningRate` | Gets or sets the learning rate. |
| `LevelSmoothing` | Gets or sets the level smoothing factor (alpha). |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension (embedding size). |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SeasonalSmoothing` | Gets or sets the seasonal smoothing factor (gamma). |
| `SequenceLength` | Gets or sets the input sequence length. |
| `TrendSmoothing` | Gets or sets the trend smoothing factor (beta). |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and returns any validation errors. |

