---
title: "SAINTNetwork<T>"
description: "SAINT (Self-Attention and Intersample Attention Transformer) neural network for tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

SAINT (Self-Attention and Intersample Attention Transformer) neural network for tabular data.

## For Beginners

SAINT is powerful because it learns two things:

Architecture:

1. **Column Attention**: Which features are related to each other?
- Self-attention within each sample across features
2. **Row Attention**: Which training samples are similar?
- Inter-sample attention comparing samples in the batch
3. **Alternating Layers**: Column and row attention alternate through the network
4. **MLP Head**: Final prediction layers

Key insight: By comparing samples within a batch, SAINT can leverage
patterns that similar samples share, making it especially effective
for semi-supervised learning and when samples have meaningful relationships.

Example flow:
Features [batch, num_features] → Embedding [batch, num_features, embed_dim]
→ Column Attention (features attend to each other)
→ Row Attention (samples attend to each other)
→ Repeat for N layers
→ MLP → Prediction

## How It Works

SAINT combines two types of attention for tabular learning:

1. Self-attention over features (column attention, like FT-Transformer)
2. Inter-sample attention (row attention, comparing samples within a batch)

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "SAINT: Improved Neural Networks for Tabular Data via Row Attention
and Contrastive Pre-Training" (2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SAINTNetwork` | Initializes a new SAINT network with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of transformer layers. |
| `Options` | Gets the SAINT-specific options. |
| `UseIntersampleAttention` | Gets whether inter-sample (row) attention is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetFeatureImportance` | Gets the learned feature importance from the attention layers. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the SAINT network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the SAINT network for the given input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the SAINT network using the provided input and expected output. |
| `UpdateNetworkParameters` | Updates the parameters of all layers in the network based on computed gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

