---
title: "FEDformerOptions<T>"
description: "Configuration options for the FEDformer (Frequency Enhanced Decomposed Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the FEDformer (Frequency Enhanced Decomposed Transformer) model.

## For Beginners

FEDformer is like listening to music - instead of processing each
sound wave individually (time domain), it analyzes the frequencies (like bass and treble).
This makes it much faster while still capturing important patterns.

Key innovations:

- **Frequency Attention:** Computes attention in frequency domain (O(n) vs O(n²))
- **Decomposition:** Separates trend (overall direction) from seasonal (repeating patterns)
- **Random Selection:** Randomly samples frequencies for efficiency

Default values are from the original FEDformer paper and work well for most datasets.

## How It Works

FEDformer achieves linear complexity by performing attention in the frequency domain
using Fourier or Wavelet transforms. It also uses seasonal-trend decomposition for
better interpretability.

**Reference:** Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
ICML 2022. https://arxiv.org/abs/2201.12740

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionType` | Gets or sets the frequency attention type. |
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward dimension. |
| `LabelLength` | Gets or sets the label length (overlap between input and prediction). |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension. |
| `MovingAverageKernel` | Gets or sets the moving average kernel size for trend extraction. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumModes` | Gets or sets the number of frequency modes to keep. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SequenceLength` | Gets or sets the input sequence length (lookback window). |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and returns any validation errors. |

