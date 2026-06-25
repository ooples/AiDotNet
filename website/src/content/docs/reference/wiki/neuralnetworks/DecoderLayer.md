---
title: "DecoderLayer<T>"
description: "Represents a Decoder Layer in a Transformer architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Decoder Layer in a Transformer architecture.

## For Beginners

The Decoder Layer helps in generating output sequences (like translations) 
by considering both what it has generated so far and the information from the input sequence.

It's like writing a story where you:

1. Look at what you've written so far (self-attention)
2. Refer back to your source material (cross-attention to encoder output)
3. Think about how to continue the story (feed-forward network)

This process helps in creating coherent and context-aware outputs.

## How It Works

The Decoder Layer is a key component in sequence-to-sequence models, particularly in Transformer architectures.
It processes the target sequence and incorporates information from the encoder output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DecoderLayer(Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the DecoderLayer class with scalar activation. |
| `DecoderLayer(Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the DecoderLayer class with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputPorts` | Declares named input ports for this multi-input layer. |
| `InputSize` | Gets the size of the input features for this layer. |
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Single-input forward pass uses the input as both decoder input and encoder output. |
| `Forward(Tensor<>[])` | Performs the forward pass of the decoder layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-resident forward pass of the decoder layer. |
| `ForwardInternal(Tensor<>,Tensor<>,Tensor<>)` | Performs the internal forward pass of the decoder layer. |
| `GetParameters` | Retrieves the current parameters of the layer. |
| `OnFirstForward(Tensor<>)` |  |
| `RequireResolvedFeedForward2(String)` | Returns `_feedForward2` if it has been constructed; throws `InvalidOperationException` with the calling member's name otherwise. |
| `ResetState` | Resets the state of the decoder layer. |
| `UpdateComponentParameters(LayerBase<>,Vector<>,Int32)` | Updates the parameters of a specific component within the decoder layer. |
| `UpdateParameters()` | Updates the layer's parameters based on the computed gradients and a learning rate. |
| `UpdateParameters(Vector<>)` | Updates the layer's parameters with the provided values. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates layer parameters using GPU-resident optimizer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_crossAttention` | The cross-attention mechanism that attends to the encoder output. |
| `_feedForward1` | The feed-forward neural network component of the decoder layer. |
| `_inputWas2D` | Tracks whether the last input was originally 2D (and thus reshaped to 3D). |
| `_lastEncoderOutput` | Stores the last encoder output tensor used by the layer. |
| `_lastEncoderOutputGradient` | Stores the gradient with respect to the encoder output from the last backward pass. |
| `_lastInput` | Stores the last input tensor processed by the layer. |
| `_lastInputGradient` | Stores the gradient with respect to the input from the last backward pass. |
| `_norm1` | Layer normalization applied after self-attention. |
| `_norm2` | Layer normalization applied after cross-attention. |
| `_norm3` | Layer normalization applied after the feed-forward network. |
| `_originalInputShape` | Stores the original input shape for restoring higher-rank tensor output. |
| `_selfAttention` | The self-attention mechanism of the decoder layer. |

