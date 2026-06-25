---
title: "TSMixerOptions<T>"
description: "Configuration options for the TSMixer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the TSMixer model.

## For Beginners

These options control how the TSMixer model behaves:

Key settings:

- **NumBlocks:** Number of mixer blocks (similar to layers in other models)
- **HiddenDimension:** Size of hidden layers in the MLPs
- **FeaturesFirst:** Whether to mix features before time (affects performance)
- **UseRevIN:** Whether to use reversible instance normalization

TSMixer is simpler than transformer-based models but can be just as effective.
It's faster to train and uses less memory than attention-based approaches.

## How It Works

TSMixer is an all-MLP architecture for multivariate time series forecasting that
achieves state-of-the-art results using only multilayer perceptrons (MLPs) without
attention mechanisms.

**Reference:** Chen et al., "TSMixer: An All-MLP Architecture for Time Series
Forecasting", TMLR 2023. https://arxiv.org/abs/2303.06053

## Properties

| Property | Summary |
|:-----|:--------|
| `Dropout` | Gets or sets the dropout rate. |
| `FeaturesFirst` | Gets or sets whether to process features before time dimension. |
| `FeedForwardExpansion` | Gets or sets the feedforward expansion factor. |
| `HiddenDimension` | Gets or sets the hidden dimension for MLP layers. |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `NumBlocks` | Gets or sets the number of mixer blocks. |
| `NumFeatures` | Gets or sets the number of input features. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SequenceLength` | Gets or sets the input sequence length. |
| `UseBatchNorm` | Gets or sets whether to use batch normalization. |
| `UseRevIN` | Gets or sets whether to use reversible instance normalization (RevIN). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and returns any validation errors. |

