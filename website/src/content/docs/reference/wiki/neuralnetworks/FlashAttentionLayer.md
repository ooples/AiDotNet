---
title: "FlashAttentionLayer<T>"
description: "A multi-head attention layer using the Flash Attention algorithm for memory-efficient computation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A multi-head attention layer using the Flash Attention algorithm for memory-efficient computation.

## For Beginners

This is like MultiHeadAttentionLayer but faster and more memory-efficient.

Flash Attention is a breakthrough algorithm that makes transformers much faster:

- Standard attention: O(N^2) memory, slow for long sequences
- Flash Attention: O(N) memory, 2-4x faster

Use this layer when:

- Training with long sequences (1024+ tokens)
- Training large models with limited GPU memory
- You need faster training/inference

The output is mathematically identical to standard attention - only the computation is different.

## How It Works

FlashAttentionLayer provides the same functionality as MultiHeadAttentionLayer but uses the
Flash Attention algorithm which is 2-4x faster and uses significantly less memory.
It can be used as a drop-in replacement in transformer architectures.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlashAttentionLayer(Int32,Int32,Int32,FlashAttentionConfig,IActivationFunction<>)` | Creates a new Flash Attention layer with the specified dimensions. |
| `FlashAttentionLayer(Int32,Int32,Int32,FlashAttentionConfig,IVectorActivationFunction<>)` | Creates a new Flash Attention layer with vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the Flash Attention configuration. |
| `HeadCount` | Gets the number of attention heads. |
| `HeadDimension` | Gets the dimension of each attention head. |
| `ParameterCount` |  |
| `PositionalEncoding` | Gets the positional encoding type used by this attention layer. |
| `RoPETheta` | Gets the RoPE base frequency (theta) if RoPE is configured. |
| `SupportsTraining` | Gets whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigurePositionalEncoding(PositionalEncodingType,Double,Int32)` | Configures positional encoding for this Flash Attention layer. |
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` | Performs the forward pass using Flash Attention. |
| `GetDiagnostics` | Gets diagnostic information about the layer. |
| `GetKeyWeights` | Gets the key projection weights. |
| `GetOutputWeights` | Gets the output projection weights. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all layer parameters as a single vector. |
| `GetQueryWeights` | Gets the query projection weights (for external access/debugging). |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetValueWeights` | Gets the value projection weights. |
| `InitializeParameters` | Initializes projection weights using Xavier/Glorot initialization. |
| `RegisterAttentionParameters` | Registers all trainable projection tensors with the layer base so that recursive parameter collection and tape-based gradient training pick them up. |
| `ResetState` | Resets the layer's internal state. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all layer parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Legacy scalar-learning-rate parameter update. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

