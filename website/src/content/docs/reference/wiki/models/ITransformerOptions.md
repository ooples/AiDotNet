---
title: "ITransformerOptions<T>"
description: "Configuration options for the iTransformer (Inverted Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the iTransformer (Inverted Transformer) model.

## For Beginners

Traditional transformers process time series by treating each time step
as a "word" (token). iTransformer flips this - it treats each variable (like price, volume, etc.)
as a token. This helps the model learn how different variables relate to each other.

Key settings:

- **NumLayers:** How many transformer layers to stack. More layers = more capacity.
- **NumHeads:** Number of attention heads. Each head learns different relationships.
- **ModelDimension:** Size of internal representations. Larger = more expressive.
- **UseChannelAttention:** If true, applies attention across channels (the key innovation).

Default values are from the original iTransformer paper and work well for most datasets.

## How It Works

iTransformer inverts the traditional transformer approach by treating each variable (channel)
as a token instead of each time step. This allows the model to learn cross-variable dependencies
more effectively.

**Reference:** Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
ICLR 2024. https://arxiv.org/abs/2310.06625

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropout` | Gets or sets the attention dropout rate. |
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward network dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension (embedding size). |
| `NumFeatures` | Gets or sets the number of input features (variables/channels). |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer encoder layers. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SequenceLength` | Gets or sets the input sequence length. |
| `UseChannelAttention` | Gets or sets whether to use channel attention (inverted attention). |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |
| `UsePreNorm` | Gets or sets whether to use pre-normalization (LayerNorm before attention). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and returns any validation errors. |

