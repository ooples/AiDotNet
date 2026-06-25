---
title: "TabPFNOptions<T>"
description: "Configuration options for TabPFN (Prior-Fitted Networks for Tabular Data)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabPFN (Prior-Fitted Networks for Tabular Data).

## For Beginners

TabPFN is unique because it learns from synthetic data:

- **Meta-learning**: Trained on millions of synthetic classification tasks
- **In-context learning**: Given training data as input, predicts test labels
- **No fine-tuning**: Works zero-shot on new datasets
- **Fast inference**: No training loop needed for new tasks

Example:

## How It Works

TabPFN is a transformer model trained on synthetic data that can perform
in-context learning on tabular classification tasks. It approximates
Bayesian inference through attention mechanisms.

Reference: "TabPFN: A Transformer That Solves Small Tabular Classification
Problems in a Second" (2022)

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoricalCardinalities` | Gets or sets the cardinalities of categorical features. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDimension` | Gets or sets the embedding dimension. |
| `FeedForwardDimension` | Gets the feed-forward dimension. |
| `FeedForwardMultiplier` | Gets or sets the feed-forward dimension multiplier. |
| `HiddenActivation` | Gets or sets the hidden layer activation function. |
| `HiddenVectorActivation` | Gets or sets the hidden layer vector activation function (alternative to scalar activation). |
| `InitScale` | Gets or sets the initialization scale. |
| `MaxClasses` | Gets or sets the maximum number of classes supported. |
| `MaxContextSamples` | Gets or sets the maximum number of training samples in context. |
| `MaxFeatures` | Gets or sets the maximum number of features supported. |
| `NumEnsembles` | Gets or sets the number of ensemble members. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `OutputHeadDimensions` | Gets or sets the hidden dimensions for the output head. |
| `UseEnsemble` | Gets or sets whether to use ensemble predictions. |
| `UsePositionalEncoding` | Gets or sets whether to use positional encoding. |
| `UsePreNorm` | Gets or sets whether to use pre-norm (norm before attention). |

