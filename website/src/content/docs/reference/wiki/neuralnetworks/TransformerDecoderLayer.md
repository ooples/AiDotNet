---
title: "TransformerDecoderLayer<T>"
description: "Represents a transformer decoder layer that processes sequences using self-attention, cross-attention, and feed-forward networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a transformer decoder layer that processes sequences using self-attention, cross-attention, and feed-forward networks.

## For Beginners

This layer helps the network generate sequences while considering both what it has generated so far and input from another source.

Think of it like a writer who is translating a book:

- First, the writer looks at what they've translated so far to maintain consistency (self-attention)
- Then they look at the original text to understand what to translate next (cross-attention)
- Finally, they process all this information to produce the next part of the translation (feed-forward network)

For example, in machine translation, the decoder generates each word of the target language by:

- Looking at the words it has already generated (to maintain grammatical coherence)
- Looking at the encoded source sentence (to understand what content to translate)
- Combining this information to produce the most appropriate next word

This architecture is powerful for tasks like translation, summarization, and text generation.

## How It Works

A transformer decoder layer is a fundamental building block of transformer-based models for sequence-to-sequence tasks.
It consists of three main components: a masked self-attention mechanism that processes the target sequence, a cross-attention
mechanism that attends to the encoder's output, and a feed-forward network for additional transformation. Each component
is followed by layer normalization and residual connections to facilitate training of deep networks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerDecoderLayer(Int32,Int32,Int32,IActivationFunction<>)` | Lazy ctor: `_embeddingSize` is resolved from `input.Shape[^1]` on first `Tensor{`; the inner attention/cross-attention/ FFN/norm sublayers are constructed then. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `InputPorts` | Declares named input ports for this multi-input layer. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer can execute on GPU. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyFeedForwardGraph(FeedForwardLayer<>,ComputationNode<>)` | Applies feed-forward graph to an input node. |
| `ApplyLayerNormGraph(LayerNormalizationLayer<>,ComputationNode<>)` | Applies layer normalization graph to an input node. |
| `ApplyMultiHeadAttentionGraph(MultiHeadAttentionLayer<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | Applies multi-head attention graph to input nodes (supports both self-attention and cross-attention). |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer by aggregating losses from sublayers. |
| `EnsureInitialized` | Constructs the sublayers using the resolved `_embeddingSize`. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Not supported for this layer. |
| `Forward(Tensor<>,Tensor<>)` | Performs the forward pass of the transformer decoder layer. |
| `ForwardGpu(Tensor<>[])` | GPU-resident forward pass for the transformer decoder layer. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetMetadata` | Persists ctor parameters that DeserializationHelper needs to reconstruct an identical layer post-clone. |
| `GetParameters` | Gets all trainable parameters of the transformer decoder layer as a single vector. |
| `OnFirstForward(Tensor<>)` | Resolves `_embeddingSize` from `input.Shape[^1]`. |
| `ResetState` | Resets the internal state of the transformer decoder layer and all its sublayers. |
| `UpdateParameters()` | Updates the parameters of all sublayers using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates layer parameters using GPU-resident optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_crossAttention` | The cross-attention mechanism for attending to the encoder output. |
| `_embeddingSize` | The size of the embeddings for queries, keys, values, and outputs. |
| `_feedForward` | The feed-forward network for additional transformation of the sequence. |
| `_feedForwardDim` | The inner dimension of the feed-forward network. |
| `_feedForwardProjection` | The projection layer that maps back from the feed-forward hidden dimension to the embedding size. |
| `_isInitialized` | True once the lazy sublayers have been constructed. |
| `_lastAuxiliaryLoss` | Stores the last computed auxiliary loss for diagnostic purposes. |
| `_lastCrossAttentionOutput` | The output tensor of the cross-attention sublayer from the last forward pass. |
| `_lastEncoderOutput` | The encoder output tensor from the last forward pass. |
| `_lastFeedForwardOutput` | The output tensor of the feed-forward sublayer from the last forward pass. |
| `_lastInput` | The input tensor from the last forward pass. |
| `_lastNormalized1` | The output tensor after the first normalization from the last forward pass. |
| `_lastNormalized2` | The output tensor after the second normalization from the last forward pass. |
| `_lastSelfAttentionOutput` | The output tensor of the self-attention sublayer from the last forward pass. |
| `_lazyFfnActivation` | FFN activation captured by the lazy ctor for later sublayer construction. |
| `_norm1` | The layer normalization applied after self-attention. |
| `_norm2` | The layer normalization applied after cross-attention. |
| `_norm3` | The layer normalization applied after the feed-forward network. |
| `_numHeads` | The number of attention heads for the self-attention and cross-attention mechanisms. |
| `_selfAttention` | The self-attention mechanism for processing the decoder input sequence. |
| `_sequenceLength` | The maximum length of the input and output sequences. |

