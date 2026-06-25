---
title: "TimesNetOptions<T>"
description: "Configuration options for the TimesNet model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the TimesNet model.

## For Beginners

These options control how the TimesNet model behaves:

Key settings:

- **TopK:** How many dominant periods to discover from the data
- **NumLayers:** How deep the network is
- **ConvKernelSize:** Size of the 2D convolution kernels
- **ModelDimension:** Size of internal representations

TimesNet is particularly good at capturing periodic patterns (daily, weekly, seasonal)
that are common in financial time series.

## How It Works

TimesNet transforms 1D time series into 2D tensors based on discovered periods,
then applies 2D convolutions to capture both intra-period and inter-period variations.

**Reference:** Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis", ICLR 2023. https://arxiv.org/abs/2210.02186

## Properties

| Property | Summary |
|:-----|:--------|
| `ConvKernelSize` | Gets or sets the 2D convolution kernel size. |
| `Dropout` | Gets or sets the dropout rate. |
| `FeedForwardDimension` | Gets or sets the feedforward network dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `LossFunction` | Gets or sets the loss function for training. |
| `ModelDimension` | Gets or sets the model dimension (embedding size). |
| `NumFeatures` | Gets or sets the number of input features. |
| `NumLayers` | Gets or sets the number of TimesBlock layers. |
| `PredictionHorizon` | Gets or sets the prediction horizon. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SequenceLength` | Gets or sets the input sequence length. |
| `TopK` | Gets or sets the number of dominant periods to discover. |
| `UseInstanceNormalization` | Gets or sets whether to use instance normalization (RevIN). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and returns any validation errors. |

