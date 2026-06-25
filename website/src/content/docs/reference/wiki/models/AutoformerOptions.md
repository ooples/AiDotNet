---
title: "AutoformerOptions<T>"
description: "Configuration options for the Autoformer model (Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Autoformer model (Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting).

## For Beginners

Autoformer is designed to capture both the trend (long-term direction)
and seasonality (repeating patterns) in time series data. Unlike Informer which focuses on attention
efficiency, Autoformer focuses on decomposing the signal into meaningful components.

Think of it like separating a song into vocals and instrumentals - by processing these separately,
the model can better understand and predict each component.

Key features:

- **Auto-Correlation**: Instead of attending to individual time points, looks at how sub-sequences

correlate with each other (like finding repeating patterns)

- **Series Decomposition**: Separates trend from seasonal patterns at every layer
- **Progressive Refinement**: Each layer further refines the decomposition

Best suited for:

- Long-horizon forecasting (weeks/months ahead)
- Data with clear seasonal patterns (energy consumption, retail sales)
- Complex trend patterns (economic indicators)

## How It Works

Autoformer (Wu et al., NeurIPS 2021) introduces a novel decomposition-based transformer architecture
for long-term time series forecasting. Key innovations include:

- Series Decomposition Block: Progressive trend-seasonal separation at each layer
- Auto-Correlation Mechanism: Efficient O(L log L) sub-series aggregation replacing self-attention
- Moving Average Kernel: Learnable trend extraction from time series

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoformerOptions` | Creates a new instance with default values. |
| `AutoformerOptions(AutoformerOptions<>)` | Creates a copy of the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoCorrelationFactor` | Gets or sets the auto-correlation aggregation factor (c in the paper). |
| `BatchSize` | Gets or sets the batch size for training. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `EmbeddingDim` | Gets or sets the embedding dimension (model width). |
| `Epochs` | Gets or sets the number of training epochs. |
| `ForecastHorizon` | Gets or sets the forecast horizon (decoder output length). |
| `LearningRate` | Gets or sets the learning rate for optimization. |
| `LookbackWindow` | Gets or sets the lookback window (encoder input length). |
| `MovingAverageKernel` | Gets or sets the kernel size for moving average in series decomposition. |
| `NumAttentionHeads` | Gets or sets the number of attention heads in auto-correlation. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |

