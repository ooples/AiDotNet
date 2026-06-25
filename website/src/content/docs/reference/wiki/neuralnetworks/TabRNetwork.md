---
title: "TabRNetwork<T>"
description: "TabR (Retrieval-Augmented) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabR (Retrieval-Augmented) neural network for tabular data.

## For Beginners

TabR is like a student who looks at similar past problems:

Architecture:

1. **Feature Encoder**: Convert features to learned embeddings
2. **Retrieval**: Find K most similar training samples
3. **Context Encoder**: Aggregate neighbor information with attention
4. **Prediction Head**: Combine query and context for prediction

Key insight: Tabular data often has local structure - similar inputs tend to
have similar outputs. TabR explicitly uses this by retrieving neighbors and
letting the model see both the query and similar training examples.

This combines the strengths of neural networks (learning features) with
k-nearest-neighbors (using local structure), often achieving SOTA results.

## How It Works

TabR combines neural networks with instance-based learning by retrieving similar
training examples and using their information to help make predictions. It encodes
both the query sample and retrieved neighbors, then aggregates the information
using attention.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabR: Tabular Deep Learning Meets Nearest Neighbors" (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabRNetwork` | Initializes a new TabR network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumLayers` | Gets the number of MLP layers. |
| `NumNeighbors` | Gets the number of neighbors to retrieve. |
| `Options` | Gets the TabR-specific options. |

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

