---
title: "TabTransformerNetwork<T>"
description: "TabTransformer neural network for tabular data with categorical features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

TabTransformer neural network for tabular data with categorical features.

## For Beginners

TabTransformer treats categorical features specially:

Architecture:

1. **Categorical Path**: Embedding → Column Embedding → Transformer → Flatten
2. **Numerical Path**: Pass through unchanged
3. **Concatenation**: Combine both paths
4. **MLP Head**: Final prediction layers

Key insight: Categorical features often have interactions that matter
(e.g., "New York" + "Finance" vs "New York" + "Farming"). The transformer
learns these relationships through self-attention.

Example flow:
Categories [batch, num_cat] → Embeddings [batch, num_cat, embed_dim]
→ Transformer [batch, num_cat, embed_dim]
→ Flatten [batch, num_cat * embed_dim]
↘
Numericals [batch, num_num] → Concat [batch, num_cat * embed_dim + num_num]
→ MLP → Prediction

## How It Works

TabTransformer applies transformer self-attention to categorical features while
passing numerical features through directly. This captures complex relationships
between categorical features that simple embeddings might miss.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabTransformerNetwork` | Initializes a new TabTransformer network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of transformer layers. |
| `Options` | Gets the TabTransformer-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetFeatureImportance` | Gets the learned feature importance from the attention layers. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the TabTransformer network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the TabTransformer network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the TabTransformer network using the provided input and expected output. |
| `UpdateNetworkParameters` | Updates the parameters of all layers in the network based on computed gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

