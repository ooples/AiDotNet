---
title: "TabDPTNetwork<T>"
description: "TabDPT (Tabular Data Pre-Training) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabDPT (Tabular Data Pre-Training) neural network for tabular data.

## For Beginners

TabDPT brings foundation model ideas to tabular data:

Architecture:

1. **Input Projection**: Map features to embedding space
2. **Transformer Encoder**: Deep self-attention for feature relationships
3. **Context Learning**: Learn from in-context examples
4. **Output Head**: Task-specific prediction layer

Key insight: By pre-training on many diverse tabular datasets,
TabDPT learns patterns that transfer to new datasets, similar to
how large language models learn from diverse text.

## How It Works

TabDPT is a foundation model approach for tabular data that uses pre-training
on diverse datasets to learn transferable representations.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabDPT: Scaling Tabular Foundation Models" (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabDPTNetwork` | Initializes a new TabDPT network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumLayers` | Gets the number of transformer layers. |
| `Options` | Gets the TabDPT-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

