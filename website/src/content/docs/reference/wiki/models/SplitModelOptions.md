---
title: "SplitModelOptions"
description: "Configuration for split neural network architecture in vertical federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for split neural network architecture in vertical federated learning.

## For Beginners

In vertical FL, a neural network is split into parts:
each party runs a "bottom model" on its local features, producing an embedding
(a compressed representation). The embeddings from all parties are combined and
fed into a "top model" that produces the final prediction.

## How It Works

This class controls how the network is split, how embeddings are sized,
and how they're combined.

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `AddEmbeddingNoise` | Gets or sets whether to add gradient noise to embeddings before sending them to the coordinator, reducing information leakage. |
| `AggregationMode` | Gets or sets how embeddings from multiple parties are combined before the top model. |
| `EmbeddingDimension` | Gets or sets the output dimension of each party's bottom model (embedding size). |
| `EmbeddingNoiseScale` | Gets or sets the standard deviation of Gaussian noise added to embeddings when `AddEmbeddingNoise` is enabled. |
| `ManualSplitLayerIndex` | Gets or sets the manual split layer index (used only when SplitPoint is Manual). |
| `SplitPoint` | Gets or sets the strategy for choosing where to split the network. |
| `TopModelHiddenDimension` | Gets or sets the hidden dimension of the top model. |
| `TopModelHiddenLayers` | Gets or sets the number of hidden layers in the top model. |
| `UseBatchNormalization` | Gets or sets whether to use batch normalization in the bottom models. |

