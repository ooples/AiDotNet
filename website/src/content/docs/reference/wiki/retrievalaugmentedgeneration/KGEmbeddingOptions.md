---
title: "KGEmbeddingOptions"
description: "Configuration options for training knowledge graph embedding models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Configuration options for training knowledge graph embedding models.

## For Beginners

These settings control how the model learns. The defaults work well
for most knowledge graphs, but you can tune them for better performance:

- EmbeddingDimension: Larger = more expressive but slower to train
- Epochs: More = better fit but risk of overfitting
- LearningRate: How fast the model updates (too high = unstable, too low = slow)
- NegativeSamples: Corrupted triples generated per real triple for contrast

## How It Works

These options control the training process for all KG embedding models (TransE, RotatE, ComplEx, DistMult).
All values use nullable types with industry-standard defaults applied internally when null.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Mini-batch size for stochastic gradient descent. |
| `EmbeddingDimension` | Dimensionality of entity and relation embedding vectors. |
| `Epochs` | Number of training epochs (full passes over all triples). |
| `L2Regularization` | L2 regularization coefficient. |
| `LearningRate` | Learning rate for SGD updates. |
| `Margin` | Margin for margin-based ranking loss (used by TransE and RotatE). |
| `NegativeSamples` | Number of negative (corrupted) samples per positive triple. |
| `NumTimeBins` | Number of time bins for temporal embedding models (TemporalTransE). |
| `Seed` | Random seed for reproducibility. |

