---
title: "NBEATSModelOptions<T>"
description: "Configuration options for the N-BEATS (Neural Basis Expansion Analysis for Time Series) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the N-BEATS (Neural Basis Expansion Analysis for Time Series) model.

## For Beginners

N-BEATS is a modern neural network approach to time series forecasting
that can automatically learn patterns from your data without requiring manual feature engineering.

Key concepts:

- Stacks: Groups of blocks that process the data hierarchically
- Blocks: Individual processing units within each stack
- Lookback Window: How many past time steps to consider for predictions
- Forecast Horizon: How many future time steps to predict
- Hidden Size: The capacity of the network (larger values can learn more complex patterns)

The model automatically decomposes your time series into interpretable components like
trend (long-term direction) and seasonality (repeating patterns).

## How It Works

N-BEATS is a deep learning architecture specifically designed for time series forecasting.
It uses a hierarchical doubly residual architecture with basis expansion to decompose
time series into trend and seasonality components, providing both accurate forecasts
and interpretability.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NBEATSModelOptions` | Initializes a new instance of the `NBEATSModelOptions` class. |
| `NBEATSModelOptions(NBEATSModelOptions<>)` | Initializes a new instance of the `NBEATSModelOptions` class by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (number of future time steps to predict). |
| `HiddenLayerSize` | Gets or sets the hidden layer size for the fully connected layers within each block. |
| `LearningRate` | Gets or sets the learning rate for training the model. |
| `LookbackWindow` | Gets or sets the lookback window size (number of historical time steps used as input). |
| `NumBlocksPerStack` | Gets or sets the number of blocks per stack. |
| `NumHiddenLayers` | Gets or sets the number of hidden layers within each block. |
| `NumStacks` | Gets or sets the number of stacks in the N-BEATS architecture. |
| `PolynomialDegree` | Gets or sets the polynomial degree for trend basis expansion. |
| `ShareWeightsInStack` | Gets or sets whether to share weights across blocks within a stack. |
| `UseInterpretableBasis` | Gets or sets whether to use interpretable basis functions (trend and seasonality). |

