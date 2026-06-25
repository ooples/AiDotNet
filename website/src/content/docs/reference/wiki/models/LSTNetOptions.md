---
title: "LSTNetOptions<T>"
description: "Configuration options for the LSTNet (Long Short-Term Time-series Network) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the LSTNet (Long Short-Term Time-series Network) model.

## For Beginners

LSTNet is like having multiple specialists working together to predict the future:

1. The convolutional part is like scanning for local patterns - like noticing "sales always spike on weekends"
2. The recurrent part remembers long-term trends - like "sales grow 10% each month"
3. The skip-RNN looks for seasonal patterns - like "Christmas sales are always highest"
4. The autoregressive part handles simple linear trends - like "each day is slightly higher than yesterday"

By combining all these, LSTNet can capture complex patterns in data where multiple time scales matter,
such as electricity consumption (hourly, daily, weekly patterns), stock prices, or traffic flow.

## How It Works

LSTNet is a neural network architecture specifically designed for multivariate time series forecasting.
It combines multiple components to capture patterns at different temporal scales:

- Convolutional layers for short-term local patterns
- Recurrent layers (GRU) for long-term dependencies
- Skip-RNN for very long periodic patterns
- Autoregressive component for local linear trends

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTNetOptions` | Initializes a new instance of the `LSTNetOptions` class with default values. |
| `LSTNetOptions(LSTNetOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoregressiveWindow` | Gets or sets the window size for the autoregressive component. |
| `BatchSize` | Gets or sets the batch size for training. |
| `ConvolutionFilters` | Gets or sets the number of convolutional filters. |
| `ConvolutionKernelSize` | Gets or sets the kernel size for convolutional layers. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HiddenRecurrentSize` | Gets or sets the hidden state size of the main recurrent layers (GRU/LSTM). |
| `HiddenSkipSize` | Gets or sets the hidden state size of the skip recurrent layers. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (context length). |
| `SkipPeriod` | Gets or sets the skip period for the Skip-RNN component. |
| `SkipRecurrentLayers` | Gets or sets the number of skip recurrent layers. |
| `UseHighway` | Gets or sets whether to use highway connections. |

