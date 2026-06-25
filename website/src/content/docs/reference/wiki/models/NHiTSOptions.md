---
title: "NHiTSOptions<T>"
description: "Configuration options for the N-HiTS (Neural Hierarchical Interpolation for Time Series) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the N-HiTS (Neural Hierarchical Interpolation for Time Series) model.

## For Beginners

N-HiTS is an advanced neural network for time series forecasting that
works by looking at your data at multiple resolutions simultaneously - similar to how you might
zoom in and out when analyzing a chart. This multi-scale approach helps it capture both
short-term patterns (like daily fluctuations) and long-term trends (like seasonal cycles).

## How It Works

N-HiTS is an evolution of N-BEATS that incorporates hierarchical interpolation and multi-rate signal sampling.
It achieves better accuracy on long-horizon forecasting tasks while being more parameter-efficient.
Key improvements over N-BEATS include:

- Hierarchical multi-rate data pooling for capturing patterns at different frequencies
- Interpolation-based basis functions for smoother forecasts
- More efficient parameter usage through stack-specific pooling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NHiTSOptions` | Initializes a new instance of the `NHiTSOptions` class. |
| `NHiTSOptions(NHiTSOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (number of future time steps to predict). |
| `HiddenLayerSize` | Gets or sets the hidden layer size for fully connected layers within each block. |
| `InterpolationModes` | Gets or sets the interpolation modes for each stack. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (number of historical time steps used as input). |
| `NumBlocksPerStack` | Gets or sets the number of blocks per stack. |
| `NumHiddenLayers` | Gets or sets the number of hidden layers within each block. |
| `NumStacks` | Gets or sets the number of stacks in the N-HiTS architecture. |
| `PoolingKernelSizes` | Gets or sets the pooling kernel sizes for each stack. |
| `PoolingModes` | Gets or sets the pooling modes for each stack. |

