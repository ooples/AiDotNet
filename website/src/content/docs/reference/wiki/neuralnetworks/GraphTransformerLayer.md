---
title: "GraphTransformerLayer<T>"
description: "Implements Graph Transformer layer using self-attention mechanisms on graph-structured data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements Graph Transformer layer using self-attention mechanisms on graph-structured data.

## For Beginners

Graph Transformers combine the power of transformers with graph structure.

Think of it like a meeting where:

- **Standard transformers**: Everyone can talk to everyone equally
- **Graph transformers**: People connected in the organizational chart get priority

Key advantages:

- Captures long-range dependencies in graphs
- More flexible than fixed neighborhood aggregation
- Can attend to any node, not just immediate neighbors
- Learns importance of connections dynamically

Use cases:

- **Large molecules**: Atoms far apart but chemically important
- **Social networks**: Identifying influential users across communities
- **Knowledge graphs**: Multi-hop reasoning
- **Program analysis**: Understanding code dependencies

Example: In a citation network, a paper can learn from:

- Direct citations (immediate neighbors)
- Indirectly related papers (through attention)
- Important papers even if not directly cited

## How It Works

Graph Transformers apply the transformer architecture to graphs by treating graph structure
as a bias in the attention mechanism. Unlike standard transformers that process sequences,
Graph Transformers incorporate graph connectivity through:

1. Structural encodings (e.g., Laplacian eigenvectors)
2. Attention biasing based on graph structure
3. Relative positional encodings for graph nodes

The attention computation is: Attention(Q, K, V) = softmax((QK^T + B)/√d_k)V
where B is a learned bias based on graph structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphTransformerLayer(Int32,Int32,Int32,Int32,Boolean,Double,IActivationFunction<>,IActivationFunction<>)` | Initializes a new instance of the `GraphTransformerLayer` class with a scalar activation function. |
| `GraphTransformerLayer(Int32,Int32,Int32,Int32,Boolean,Double,IVectorActivationFunction<>,IActivationFunction<>)` | Initializes a new instance of the `GraphTransformerLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` |  |
| `OutputFeatures` |  |
| `ParameterCount` |  |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFFNActivation(Tensor<>)` | Applies the configured FFN activation function to the input tensor. |
| `ApplyGraphMaskGpu(IDirectGpuBackend,IGpuBuffer,Tensor<>,Int32,Int32)` | Applies graph mask to attention scores for 3D adjacency matrices with per-batch masking. |
| `ClearGradients` |  |
| `CopyHeadToConcat(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,Int32,Int32,Int32)` | Copies a head's output to the correct position in the concatenated buffer. |
| `ExtractHeadWeightsFloat(Int32,Tensor<>)` | Extracts weight matrix for a specific head as a float array. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass for GraphTransformerLayer. |
| `GetAdjacencyMatrix` |  |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `InitializeWeightTensors` | Initializes weight tensors with appropriate shapes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `ScatterHeadSlice(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Writes a per-head [batch, rows, cols] tensor into the h-th slice of a 4D destination [batch, numHeads, rows, cols]. |
| `Serialize(BinaryWriter)` |  |
| `SetAdjacencyMatrix(Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_adjacencyMatrix` | The adjacency matrix defining graph structure. |
| `_ffnActivation` | Activation function for the feed-forward network hidden layer. |
| `_ffnBias1` | Feed-forward network first layer bias: [ffnHiddenDim] |
| `_ffnBias2` | Feed-forward network second layer bias: [outputFeatures] |
| `_ffnWeights1` | Feed-forward network first layer weights: [outputFeatures, ffnHiddenDim] |
| `_ffnWeights2` | Feed-forward network second layer weights: [ffnHiddenDim, outputFeatures] |
| `_keyWeights` | Key transformation weights for each head: [numHeads, inputFeatures, headDim] |
| `_lastInput` | Cached values for backward pass. |
| `_layerNorm1Bias` | Layer normalization bias for first norm: [outputFeatures] |
| `_layerNorm1Scale` | Layer normalization scale for first norm: [outputFeatures] |
| `_layerNorm2Bias` | Layer normalization bias for second norm: [outputFeatures] |
| `_layerNorm2Scale` | Layer normalization scale for second norm: [outputFeatures] |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outputBias` | Output projection bias: [outputFeatures] |
| `_outputWeights` | Output projection weights: [numHeads * headDim, outputFeatures] |
| `_queryWeights` | Query transformation weights for each head: [numHeads, inputFeatures, headDim] |
| `_queryWeightsGradient` | Gradients for parameters. |
| `_structuralBias` | Structural bias for attention (learned from graph structure): [numHeads, maxNodes, maxNodes] |
| `_valueWeights` | Value transformation weights for each head: [numHeads, inputFeatures, headDim] |

