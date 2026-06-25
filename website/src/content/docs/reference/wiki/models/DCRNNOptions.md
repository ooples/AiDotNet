---
title: "DCRNNOptions<T>"
description: "Configuration options for DCRNN (Diffusion Convolutional Recurrent Neural Network)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for DCRNN (Diffusion Convolutional Recurrent Neural Network).

## For Beginners

DCRNN was specifically designed for traffic forecasting
by combining two powerful ideas:

**The Key Insight:**
Traffic flow on road networks can be modeled as a diffusion process - like how
congestion spreads through a network. DCRNN captures this with diffusion convolution
while using an encoder-decoder architecture for multi-step prediction.

**What Problems Does DCRNN Solve?**

- Traffic speed/flow prediction on road networks
- Air quality forecasting across sensor networks
- Subway ridership prediction
- Any spatial-temporal forecasting where diffusion dynamics matter

**How DCRNN Works:**

1. **Diffusion Convolution:** Models spatial dependencies as bidirectional random walks
2. **Diffusion GRU:** Replaces matrix multiplications in GRU with diffusion convolution
3. **Encoder-Decoder:** Encoder captures history, decoder generates predictions
4. **Scheduled Sampling:** Gradually transitions from ground truth to predictions during training

**DCRNN Architecture:**

- Encoder: Stacked DCGRU layers that encode input sequence
- Decoder: Stacked DCGRU layers that generate output sequence
- Diffusion: D_O^(K) + D_I^(K) bidirectional diffusion matrices
- Output: Linear projection to forecast values

**Key Benefits:**

- Captures spatial dependencies through diffusion process (not just adjacency)
- Multi-step prediction through encoder-decoder architecture
- Scheduled sampling prevents exposure bias during training
- Bidirectional diffusion captures both upstream and downstream effects

## How It Works

DCRNN combines diffusion convolution with sequence-to-sequence architecture
for spatial-temporal forecasting on graph-structured data.

**Reference:** Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.
https://arxiv.org/abs/1707.01926

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DCRNNOptions` | Initializes a new instance of the `DCRNNOptions` class with default values. |
| `DCRNNOptions(DCRNNOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionSteps` | Gets or sets the number of diffusion steps (K). |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FilterType` | Gets or sets the type of filter for diffusion. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension for the DCGRU cells. |
| `MaxDiffusionStep` | Gets or sets the maximum diffusion step power. |
| `MinTeacherForcingRatio` | Gets or sets the minimum teacher forcing ratio. |
| `NumDecoderLayers` | Gets or sets the number of decoder DCGRU layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder DCGRU layers. |
| `NumFeatures` | Gets or sets the number of input features per node. |
| `NumNodes` | Gets or sets the number of nodes in the graph. |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `ScheduledSamplingDecaySteps` | Gets or sets the number of training steps for scheduled sampling decay. |
| `SequenceLength` | Gets or sets the sequence length (input time steps). |
| `UseScheduledSampling` | Gets or sets whether to use scheduled sampling during training. |

