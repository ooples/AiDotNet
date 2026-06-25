---
title: "TransformerEncoderLayer<T>"
description: "Represents a transformer encoder layer that processes sequences using self-attention and feed-forward networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a transformer encoder layer that processes sequences using self-attention and feed-forward networks.

## For Beginners

This layer helps a neural network understand relationships between different elements in a sequence.

Think of it like a careful reader analyzing a paragraph:

- First, the reader looks at how each word relates to every other word (self-attention)
- Then, the reader processes this information to understand the meaning (feed-forward network)

For example, in the sentence "The animal didn't cross the street because it was too wide":

- The self-attention helps the network understand that "it" refers to "the street" (not "the animal")
- The feed-forward network processes this contextual information for each word

This architecture is powerful for tasks like understanding text, analyzing time series, or processing any data
where the relationships between elements matter.

## How It Works

A transformer encoder layer is a fundamental building block of transformer-based models for sequence processing tasks.
It consists of two main components: a self-attention mechanism that allows each position in a sequence to attend to all
positions, and a feed-forward network that processes each position independently. Each component is followed by layer
normalization and residual connections to facilitate training of deep networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerEncoderLayer(Int32,Int32)` | Lazy ctor: `_embeddingSize` is resolved from `input.Shape[^1]` on first `Tensor{`; the inner attention/FFN/norm sublayers are constructed then. |
| `TransformerEncoderLayer(Int32,Int32,Int32)` | Eager-dimension ctor: `_embeddingSize` is known at construction time, so `ParameterCount` can declare a non-zero count immediately (no first-forward required). |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFeedForwardGraph(FeedForwardLayer<>,ComputationNode<>)` | Applies feed-forward graph to an input node. |
| `ApplyLayerNormGraph(LayerNormalizationLayer<>,ComputationNode<>)` | Applies layer normalization graph to an input node. |
| `ApplyMultiHeadAttentionGraph(MultiHeadAttentionLayer<>,ComputationNode<>)` | Applies multi-head attention graph to an input node. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer by aggregating losses from sublayers. |
| `EnsureInitialized` | Constructs the sublayers (attention, norms, FFN) using the resolved `_embeddingSize`. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetMetadata` | Returns layer-specific metadata for serialization. |
| `GetParameters` | Gets all trainable parameters of the transformer encoder layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves `_embeddingSize` from `input.Shape[^1]` and propagates the full input shape into the layer's resolved shapes (input == output for an encoder block). |
| `ResetState` | Resets the internal state of the transformer encoder layer and all its sublayers. |
| `TryDeclareShape` | AiDotNet#1370 shape oracle override: when the eager-dimension ctor (`(numHeads, feedForwardDim, embeddingSize)`) supplied a concrete embedding width, sublayers were constructed at ctor time (`EnsureInitialized` ran from the eager ctor). |
| `UpdateParameters()` | Updates the parameters of all sublayers using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates layer parameters using GPU-resident optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingSize` | The size of the embeddings for queries, keys, values, and outputs. |
| `_feedForward1` | The feed-forward network for additional transformation of the sequence. |
| `_feedForward2` | The second (projection) layer of the feed-forward network. |
| `_feedForwardDim` | The inner dimension of the feed-forward network. |
| `_inputWas2D` | Tracks whether the last input was originally 2D (and thus reshaped to 3D). |
| `_isInitialized` | True once sublayers (attention, norm1, ffn1, ffn2, norm2) have been constructed. |
| `_lastAuxiliaryLoss` | Stores the last computed auxiliary loss for diagnostic purposes. |
| `_norm1` | The layer normalization applied after self-attention. |
| `_norm2` | The layer normalization applied after the feed-forward network. |
| `_numHeads` | The number of attention heads for the self-attention mechanism. |
| `_originalInputShape` | Stores the original input shape for restoring higher-rank tensor output. |
| `_selfAttention` | The self-attention mechanism for processing the input sequence. |

