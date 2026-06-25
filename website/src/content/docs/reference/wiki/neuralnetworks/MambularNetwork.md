---
title: "MambularNetwork<T>"
description: "Mambular (State Space Model for Tabular Data) neural network."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Mambular (State Space Model for Tabular Data) neural network.

## For Beginners

Mambular treats features like a sequence:

Architecture:

1. **Feature Embedding**: Convert features to learned representations
2. **State Space Layers**: Process features sequentially with memory
3. **Selective Mechanism**: Learn which features to remember/forget
4. **MLP Head**: Final prediction from processed features

Key insight: State space models (like Mamba) are more efficient than
transformers for long sequences. For tabular data with many features,
this can provide both better scaling and learned sequential relationships.

## How It Works

Mambular applies the Mamba state space model architecture to tabular data,
treating features as a sequence and using selective state spaces for processing.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Mambular: A Sequential Model for Tabular Deep Learning" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MambularNetwork` | Initializes a new instance with default architecture settings. |
| `MambularNetwork(NeuralNetworkArchitecture<>,MambularOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new Mambular network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumLayers` | Gets the number of layers. |
| `Options` | Gets the Mambular-specific options. |
| `StateDimension` | Gets the state dimension for the SSM. |

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

