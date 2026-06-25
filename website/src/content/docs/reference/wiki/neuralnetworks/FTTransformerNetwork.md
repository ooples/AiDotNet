---
title: "FTTransformerNetwork<T>"
description: "FT-Transformer (Feature Tokenizer + Transformer) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

FT-Transformer (Feature Tokenizer + Transformer) neural network for tabular data.

## For Beginners

FT-Transformer brings NLP's Transformer to tables:

Architecture:

1. **Feature Tokenization**: Each column becomes a "token" (like words in text)
2. **[CLS] Token**: Special token to aggregate information for prediction
3. **Transformer Encoder**: Self-attention learns feature relationships
4. **Classification Head**: MLP on [CLS] token for final prediction

Key insight: Just as transformers revolutionized NLP by learning word relationships,
FT-Transformer learns feature relationships through attention. Each feature can
"attend" to other features to capture complex interactions.

This often outperforms gradient boosting on larger datasets and automatically
learns which feature combinations matter.

## How It Works

FT-Transformer applies the transformer architecture to tabular data by treating each feature
as a token. It tokenizes numerical and categorical features into embeddings and processes
them with standard transformer encoder layers.
This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FTTransformerNetwork` | Initializes a new FT-Transformer network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of transformer layers. |
| `Options` | Gets the FT-Transformer-specific options. |

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

