---
title: "CrossformerOptions<T>"
description: "Configuration options for the Crossformer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Crossformer model.

## For Beginners

These options control how the Crossformer model behaves:

Key settings:

- **SegmentLength:** How long each time segment is for cross-time attention.
- **NumLayers:** How deep the transformer is. More layers = more capacity but slower.
- **NumHeads:** Attention heads. More heads can capture different patterns.
- **RouterTopK:** For mixture of experts, how many experts to use per token.

Crossformer excels at capturing both temporal patterns and cross-variable relationships
that are common in multivariate financial time series.

## How It Works

Crossformer uses a cross-dimension attention mechanism that captures both temporal
and cross-variable dependencies simultaneously through a two-stage attention structure.

**Reference:** Zhang et al., "Crossformer: Transformer Utilizing Cross-Dimension
Dependency for Multivariate Time Series Forecasting", ICLR 2023.
https://openreview.net/forum?id=vSVLM2j9eie

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropout` | Gets or sets the attention dropout rate. |
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward network dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension (embedding size). |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer encoder layers. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `RouterTopK` | Gets or sets the top-K value for router in cross-dimension attention. |
| `SegmentLength` | Gets or sets the segment length for cross-time attention. |
| `SequenceLength` | Gets or sets the input sequence length. |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |
| `UsePreNorm` | Gets or sets whether to use pre-norm (LayerNorm before attention/FFN). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateNumSegments` | Calculates the number of segments based on sequence length and segment length. |
| `Validate` | Validates the options and returns any validation errors. |

