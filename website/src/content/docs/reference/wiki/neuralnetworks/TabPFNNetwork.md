---
title: "TabPFNNetwork<T>"
description: "TabPFN (Prior-Fitted Network) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabPFN (Prior-Fitted Network) neural network for tabular data.

## For Beginners

TabPFN is unique because it learns from synthetic data:

Architecture:

1. **Input Projection**: Convert features to embeddings
2. **Deep Transformer**: Many attention layers (typically 12)
3. **In-Context Learning**: Training data provided as context
4. **Output Prediction**: Classification directly from attention

Key insight: TabPFN was trained on millions of synthetic classification
tasks, learning to "learn" from examples. At inference, you provide
training data as context and it predicts test labels without fine-tuning.

This makes TabPFN extremely fast for small datasets since there's no
traditional training loop - just a forward pass with context.

## How It Works

TabPFN is a transformer model trained on synthetic data that can perform
in-context learning on tabular classification tasks. It approximates
Bayesian inference through attention mechanisms.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second" (2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabPFNNetwork` | Initializes a new instance with default architecture settings. |
| `TabPFNNetwork(NeuralNetworkArchitecture<>,TabPFNOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new TabPFN network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumLayers` | Gets the number of transformer layers. |
| `Options` | Gets the TabPFN-specific options. |

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

