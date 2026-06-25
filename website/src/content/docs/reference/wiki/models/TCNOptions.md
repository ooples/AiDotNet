---
title: "TCNOptions<T>"
description: "Configuration options for the TCN (Temporal Convolutional Network) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the TCN (Temporal Convolutional Network) model.

## For Beginners

TCN is a powerful alternative to LSTM/GRU for sequence modeling:

**Key Concepts:**

- **Causal Convolutions:** Each prediction only depends on past values, never future ones

(important for real-time prediction)

- **Dilated Convolutions:** Instead of looking at consecutive time steps, TCN skips steps

with increasing gaps (dilation). With dilations [1, 2, 4, 8], the network can "see" far
into the past without needing huge filters.

**Example:**

- Layer 1 (dilation=1): Looks at times [t-2, t-1, t]
- Layer 2 (dilation=2): Looks at times [t-4, t-2, t]
- Layer 3 (dilation=4): Looks at times [t-8, t-4, t]
- Together: Can see 14 time steps back with only 3 layers!

**Benefits:**

- Parallelizable (faster training than RNNs)
- Flexible receptive field (controls how far back to look)
- No vanishing gradient problem
- Good for long sequences

## How It Works

TCN uses dilated causal convolutions to model temporal sequences. Unlike recurrent networks,
TCN processes sequences in parallel, making it faster to train while still capturing
long-range dependencies through its exponentially increasing dilation factors.

**Reference:** Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent
Networks for Sequence Modeling", 2018. https://arxiv.org/abs/1803.01271

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TCNOptions` | Initializes a new instance of the `TCNOptions` class with default values. |
| `TCNOptions(TCNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `KernelSize` | Gets or sets the kernel size for convolutional layers. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (input sequence length). |
| `NumChannels` | Gets or sets the number of channels (filters) in each convolutional layer. |
| `NumLayers` | Gets or sets the number of TCN layers. |
| `UseResidualConnections` | Gets or sets whether to use residual connections. |

