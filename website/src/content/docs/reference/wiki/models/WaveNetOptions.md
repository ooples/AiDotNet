---
title: "WaveNetOptions<T>"
description: "Configuration options for the WaveNet model adapted for time series forecasting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the WaveNet model adapted for time series forecasting.

## For Beginners

WaveNet is similar to TCN but with some key differences:

**Gated Activation Units:**
Instead of simple ReLU activations, WaveNet uses gates:

- tanh(Wf * x) * sigmoid(Wg * x)
- The sigmoid acts as a "gate" controlling information flow
- This helps model complex patterns more effectively

**Skip Connections:**
WaveNet has TWO types of connections:

1. **Residual:** Connect input to output of each block (like TCN)
2. **Skip:** Each block also sends output directly to the final layers
- This allows the network to combine features from different time scales

**Stacked Dilations:**
WaveNet often repeats the dilation pattern multiple times:

- [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, ...]
- This creates very deep networks with huge receptive fields

Originally designed for generating audio one sample at a time, WaveNet's architecture
is now used for many sequence prediction tasks including financial forecasting.

## How It Works

WaveNet was originally developed by DeepMind for audio generation but has proven highly effective
for time series forecasting. It uses dilated causal convolutions with gated activations and
residual/skip connections.

**Reference:** van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016.
https://arxiv.org/abs/1609.03499

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveNetOptions` | Initializes a new instance of the `WaveNetOptions` class with default values. |
| `WaveNetOptions(WaveNetOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for training. |
| `DilationDepth` | Gets or sets the dilation depth (number of dilation doublings per stack). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `KernelSize` | Gets or sets the kernel size for dilated convolutions. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `LookbackWindow` | Gets or sets the lookback window size (input sequence length). |
| `NumStacks` | Gets or sets the number of stacks (repetitions of the dilation pattern). |
| `ResidualChannels` | Gets or sets the number of residual channels. |
| `SkipChannels` | Gets or sets the number of skip channels. |
| `UseGatedActivations` | Gets or sets whether to use gated activation units. |

