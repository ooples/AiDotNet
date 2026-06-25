---
title: "MQCNNOptions<T>"
description: "Configuration options for the MQCNN (Multi-Quantile Convolutional Neural Network) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the MQCNN (Multi-Quantile Convolutional Neural Network) model.

## For Beginners

MQCNN combines CNNs with quantile regression for forecasting:

**What is Quantile Forecasting?**
Instead of predicting a single value, MQCNN predicts a range:

- The 10th percentile (P10): "90% of actual values will be above this"
- The 50th percentile (P50): The median prediction
- The 90th percentile (P90): "90% of actual values will be below this"

**Why Multiple Quantiles?**

- Captures uncertainty in predictions
- Provides prediction intervals, not just point estimates
- Useful for risk management (worst-case/best-case scenarios)
- Better decision making with confidence bounds

**Example:**
For tomorrow's stock price, instead of "100.50", you get:

- P10: 98.20 (likely floor)
- P50: 100.50 (median)
- P90: 102.80 (likely ceiling)

**Architecture:**

1. **Encoder:** Dilated causal convolutions process the input sequence
2. **Context:** Extracted features represent temporal patterns
3. **Decoder:** Separate output heads for each quantile
4. **Loss:** Quantile loss (pinball loss) for each quantile level

## How It Works

MQCNN is a probabilistic forecasting model that predicts multiple quantiles simultaneously,
providing uncertainty estimates along with point forecasts. It uses dilated causal convolutions
to model temporal dependencies and outputs predictions at multiple quantile levels.

**Reference:** Wen et al., "A Multi-Horizon Quantile Recurrent Forecaster", 2017.
https://arxiv.org/abs/1711.11053

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MQCNNOptions` | Initializes a new instance of the `MQCNNOptions` class with default values. |
| `MQCNNOptions(MQCNNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DecoderChannels` | Gets or sets the number of channels in the decoder network. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EncoderChannels` | Gets or sets the number of channels in the encoder network. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `KernelSize` | Gets or sets the kernel size for convolutional layers. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (input sequence length). |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `Quantiles` | Gets or sets the quantile levels to predict. |

