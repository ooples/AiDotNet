---
title: "TemporalFusionTransformerOptions<T>"
description: "Configuration options for the Temporal Fusion Transformer (TFT) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Temporal Fusion Transformer (TFT) model.

## For Beginners

TFT is an advanced neural network designed specifically for forecasting
that can handle multiple types of input data:

- Static features (e.g., store location, product category) that don't change over time
- Known future inputs (e.g., holidays, promotions) that we know ahead of time
- Unknown inputs (e.g., past sales) that we can only observe historically

The model uses "attention" mechanisms to focus on the most relevant time periods and features,
making it both accurate and interpretable.

## How It Works

Temporal Fusion Transformer is a state-of-the-art deep learning architecture for multi-horizon forecasting.
It combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics.
TFT uses self-attention mechanisms to learn temporal relationships at different scales and integrates
static metadata, time-varying known inputs, and time-varying unknown inputs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemporalFusionTransformerOptions` | Initializes a new instance of the `TemporalFusionTransformerOptions` class. |
| `TemporalFusionTransformerOptions(TemporalFusionTransformerOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (number of future time steps to predict). |
| `HiddenSize` | Gets or sets the hidden state size for the model. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (number of historical time steps used as input). |
| `NumAttentionHeads` | Gets or sets the number of attention heads in the multi-head attention mechanism. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `QuantileLevels` | Gets or sets the quantile levels for probabilistic forecasting. |
| `StaticCovariateSize` | Gets or sets the number of static covariates (features that don't change over time). |
| `TimeVaryingKnownSize` | Gets or sets the number of time-varying known inputs (future values that are known). |
| `TimeVaryingUnknownSize` | Gets or sets the number of time-varying unknown inputs (past observations only). |
| `UseVariableSelection` | Gets or sets whether to use variable selection networks. |

