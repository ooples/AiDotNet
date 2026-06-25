---
title: "TabMOptions<T>"
description: "Configuration options for TabM, a parameter-efficient ensemble model for tabular data."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TabM, a parameter-efficient ensemble model for tabular data.

## For Beginners

TabM is a smart way to get the benefits of model ensembles
(multiple models voting together) without the huge computational cost.

Key concepts:

- **Ensembles**: Multiple models combined usually give better predictions
- **Parameter Sharing**: TabM shares most weights across ensemble members
- **Rank Vectors**: Small per-member vectors that customize the shared weights
- **Efficient**: Gets ensemble benefits with minimal extra parameters

Why this matters:

- Traditional ensembles need N times the parameters for N models
- TabM only adds about 1-5% extra parameters per ensemble member
- Same computational cost as a single model during inference
- Often outperforms both single models and traditional ensembles

Example usage:

## How It Works

TabM uses BatchEnsemble-style parameter sharing to create multiple ensemble members
with minimal parameter overhead. Each member shares base weights but has its own
small rank vectors that modulate the shared weights.

Reference: "TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling" (2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `ActivationType` | Gets or sets the activation function type. |
| `AverageEnsemble` | Gets or sets whether to average ensemble predictions or concatenate. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EnableGradientClipping` | Gets or sets whether to enable gradient clipping. |
| `FeatureEmbeddingDimension` | Gets or sets the embedding dimension for feature embeddings. |
| `HiddenDimensions` | Gets or sets the hidden layer dimensions. |
| `MaxGradientNorm` | Gets or sets the maximum gradient norm for clipping. |
| `NumEnsembleMembers` | Gets or sets the number of ensemble members. |
| `RankInitScale` | Gets or sets the initialization scale for rank vectors. |
| `UseBias` | Gets or sets whether to use bias in the ensemble layers. |
| `UseFeatureEmbeddings` | Gets or sets whether to use numerical feature embeddings. |
| `UseLayerNorm` | Gets or sets whether to use layer normalization. |
| `WeightDecay` | Gets or sets the weight decay (L2 regularization) coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a copy of the options. |

