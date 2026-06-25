---
title: "PatchTSTOptions<T>"
description: "Configuration options for the PatchTST (Patch Time Series Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the PatchTST (Patch Time Series Transformer) model.

## For Beginners

These options control how the PatchTST model behaves:

Key settings:

- **PatchSize:** How long each "patch" (segment) of time series is.

Smaller patches capture fine details; larger patches capture broader patterns.

- **Stride:** How much overlap between patches. Equal to PatchSize for no overlap.
- **NumLayers:** How deep the transformer is. More layers = more capacity but slower.
- **NumHeads:** Attention heads. More heads can capture different types of patterns.
- **ChannelIndependent:** Whether to process each variable independently.

Usually True works better for multivariate forecasting.

Default values are from the original PatchTST paper and work well for most datasets.

## How It Works

PatchTST divides time series into patches and processes them with a Transformer encoder.
This approach has shown state-of-the-art results on long-term forecasting benchmarks.

**Reference:** Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropout` | Gets or sets the attention dropout rate. |
| `ChannelIndependent` | Gets or sets whether to use channel-independent (CI) mode. |
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward network dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension (embedding size). |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer encoder layers. |
| `PatchSize` | Gets or sets the patch size (segment length). |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SequenceLength` | Gets or sets the input sequence length. |
| `Stride` | Gets or sets the stride between consecutive patches. |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |
| `UsePreNorm` | Gets or sets whether to use pre-norm (LayerNorm before attention/FFN) or post-norm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateNumPatches` | Calculates the number of patches based on sequence length, patch size, and stride. |
| `Validate` | Validates the options and returns any validation errors. |

