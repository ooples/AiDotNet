---
title: "TabRBase<T>"
description: "Base implementation of TabR, a retrieval-augmented model for tabular data."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NeuralNetworks.Tabular`

Base implementation of TabR, a retrieval-augmented model for tabular data.

## For Beginners

TabR is like having a photographic memory for training data.

Architecture overview:

1. **Feature Encoder**: MLP that converts raw features to embeddings
2. **Retrieval Index**: Stores embeddings of all training samples
3. **K-NN Search**: Finds the K most similar training samples
4. **Context Encoder**: Uses attention to aggregate neighbor information
5. **Prediction Head**: Makes final prediction using combined information

Why this approach works:

- Tabular data often has "local" structure (similar inputs → similar outputs)
- Neural networks alone may struggle with rare patterns
- Retrieval provides explicit "memory" of similar past examples
- Attention learns which neighbors are most relevant

Think of it as:

- Traditional k-NN: Just average neighbors (simple but limited)
- TabR: Learn to encode, then learn to attend to neighbors (powerful)

## How It Works

TabR combines neural networks with instance-based learning. It encodes features,
retrieves similar training samples, and uses attention to aggregate neighbor
information for making predictions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabRBase(Int32,TabROptions<>)` | Initializes a new instance of the TabRBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `IsIndexBuilt` | Gets whether the retrieval index has been built. |
| `NumIndexedSamples` | Gets the number of samples in the retrieval index. |
| `NumNeighbors` | Gets the number of nearest neighbors to retrieve. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateNeighbors(Tensor<>,Tensor<>)` | Aggregates neighbor information using attention. |
| `BuildIndex(Tensor<>)` | Builds the retrieval index from training data. |
| `EncodeFeatures(Tensor<>)` | Encodes input features to embeddings. |
| `ForwardBackbone(Tensor<>,Vector<Int32>)` | Performs the forward pass through the TabR backbone. |
| `GetAttentionWeights` | Gets the attention weights from the last forward pass. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetRetrievedNeighborIndices` | Gets the retrieved neighbor indices from the last forward pass. |
| `NormalizeEmbeddings(Tensor<>)` | Normalizes embeddings to unit length. |
| `ResetState` | Resets internal state. |
| `RetrieveNeighbors(Tensor<>,Vector<Int32>)` | Retrieves the K nearest neighbors for query samples. |
| `UpdateParameters()` | Updates all parameters using the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumFeatures` | Number of input features. |
| `NumOps` | Numeric operations helper for type T. |
| `Options` | The model configuration options. |

