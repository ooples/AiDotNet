---
title: "FTTransformerBase<T>"
description: "Base implementation of FT-Transformer (Feature Tokenizer + Transformer) for tabular data."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base implementation of FT-Transformer (Feature Tokenizer + Transformer) for tabular data.

## For Beginners

FT-Transformer brings the power of the Transformer architecture (used in
GPT, BERT, etc.) to traditional tabular data like spreadsheets and databases.

Architecture overview:

1. **Feature Tokenizer**: Converts each column value into a vector embedding
2. **[CLS] Token**: A special learnable token prepended to capture global information
3. **Transformer Layers**: Self-attention layers to capture feature interactions
4. **Prediction Head**: Uses the [CLS] token output to make predictions

How it works:

- Each feature (column) becomes a "token" with its own embedding
- Self-attention allows any feature to "look at" any other feature
- The [CLS] token aggregates information from all features
- Final prediction is made based on the [CLS] representation

Key advantages:

- Learns which features interact with each other automatically
- No manual feature engineering needed
- Often outperforms gradient boosting on larger datasets
- Attention weights provide interpretability

## How It Works

FT-Transformer is a simple but effective adaptation of the Transformer architecture for tabular data.
It treats each feature as a token by embedding it into a d-dimensional vector, then processes
the sequence with standard transformer encoder layers.

Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FTTransformerBase(Int32,FTTransformerOptions<>)` | Initializes a new instance of the FTTransformerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `NumLayers` | Gets the number of transformer layers. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SequenceLength` | Gets the sequence length including [CLS] token. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the FT-Transformer backbone. |
| `GetAttentionWeights` | Gets attention weights from all layers for interpretability. |
| `GetFeatureImportance` | Computes feature importance based on the attention patterns. |
| `GetParameterGradients` | Gets parameter gradients as a single vector. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `ResetState` | Resets internal state including caches and gradients. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a vector. |
| `SetTrainingMode(Boolean)` | Sets the training mode for the model. |
| `UpdateParameters()` | Updates all parameters using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `EncoderLayers` | The transformer encoder layers. |
| `FinalLayerNorm` | Final layer normalization applied after the transformer. |
| `NumCategoricalFeatures` | Number of categorical features. |
| `NumFeatures` | Number of input features (numerical + categorical). |
| `NumNumericalFeatures` | Number of numerical features. |
| `NumOps` | Numeric operations helper for type T. |
| `Options` | The model configuration options. |
| `Tokenizer` | The feature tokenizer that converts features to embeddings. |

