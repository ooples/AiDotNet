---
title: "TabTransformerBase<T>"
description: "Base implementation of TabTransformer for tabular data."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base implementation of TabTransformer for tabular data.

## For Beginners

TabTransformer is like FT-Transformer but treats categories specially:

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabTransformerBase(Int32,TabTransformerOptions<>)` | Initializes a new instance of the TabTransformerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CombinedDimension` | Gets the combined feature dimension after concatenation. |
| `EmbeddingDimension` | Gets the embedding dimension. |
| `Engine` | Hardware-accelerated engine for tensor operations. |
| `MLPOutputDimension` | Gets the MLP output dimension (last hidden dimension or combined if no MLP layers). |
| `NumLayers` | Gets the number of transformer layers. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedCategoricals(Matrix<Int32>)` | Embeds categorical features. |
| `ForwardBackbone(Tensor<>,Matrix<Int32>)` | Performs the forward pass through the TabTransformer backbone. |
| `InitializeNormal(Tensor<>,Double,Random)` | Initializes a tensor with normal distribution. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates all parameters using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumCategoricalFeatures` | Number of categorical features. |
| `NumNumericalFeatures` | Number of numerical features. |
| `NumOps` | Numeric operations helper for type T. |
| `Options` | The model configuration options. |

