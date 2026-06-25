---
title: "AutoIntNetwork<T>"
description: "AutoInt (Automatic Feature Interaction Learning) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

AutoInt (Automatic Feature Interaction Learning) neural network for tabular data.

## For Beginners

AutoInt discovers feature interactions automatically:

Architecture:

1. **Feature Embeddings**: Each feature gets a learned vector representation
2. **Self-Attention Layers**: Features attend to each other to learn interactions
3. **Residual Connections**: Preserve individual feature information
4. **MLP Head**: Final prediction from interaction-enhanced features

Key insight: Feature interactions (e.g., "age + income" or "city + job")
are often important for predictions. AutoInt learns these automatically
through attention, capturing which features should be combined.

Example: In click prediction, "user_age + product_category" might have
a strong interaction that AutoInt discovers without manual feature engineering.

## How It Works

AutoInt uses multi-head self-attention to automatically learn high-order
feature interactions without manual feature engineering.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" (2018)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoIntNetwork` | Initializes a new AutoInt network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of interacting layers. |
| `Options` | Gets the AutoInt-specific options. |

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

