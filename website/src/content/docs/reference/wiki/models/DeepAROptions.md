---
title: "DeepAROptions<T>"
description: "Configuration options for the DeepAR (Deep Autoregressive) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the DeepAR (Deep Autoregressive) model.

## For Beginners

DeepAR is an advanced forecasting model that not only predicts
what will happen, but also how confident it is in those predictions. Instead of saying
"sales will be exactly 100 units," it might say "sales will likely be between 80 and 120 units,
with 100 being most probable."

This is especially useful when:

- You need to plan for worst-case and best-case scenarios
- You have many related time series (e.g., sales across many stores)
- You have some series with very little historical data

The "autoregressive" part means it uses its own predictions as inputs for future predictions,
and "deep" refers to the use of deep neural networks (specifically, LSTM networks).

## How It Works

DeepAR is a probabilistic forecasting methodology based on autoregressive recurrent neural networks.
Unlike traditional methods that provide point forecasts, DeepAR produces probabilistic forecasts
that include prediction intervals. It's particularly effective for:

- Handling multiple related time series simultaneously
- Cold-start problems (forecasting for new items with limited history)
- Capturing complex seasonal patterns and trends
- Quantifying forecast uncertainty

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepAROptions` | Initializes a new instance of the `DeepAROptions` class. |
| `DeepAROptions(DeepAROptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `CovariateSize` | Gets or sets the number of covariates (external features). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for categorical features. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenSize` | Gets or sets the hidden state size of the LSTM layers. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LikelihoodType` | Gets or sets the likelihood distribution type. |
| `LookbackWindow` | Gets or sets the lookback window size (context length). |
| `NumLayers` | Gets or sets the number of LSTM layers. |
| `NumSamples` | Gets or sets the number of samples to draw for probabilistic forecasts. |

