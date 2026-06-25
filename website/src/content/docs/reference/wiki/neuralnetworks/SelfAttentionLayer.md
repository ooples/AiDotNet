---
title: "SelfAttentionLayer<T>"
description: "Represents a self-attention layer that allows a sequence to attend to itself, capturing relationships between elements."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a self-attention layer that allows a sequence to attend to itself, capturing relationships between elements.

## For Beginners

This layer helps a neural network understand relationships between different parts of a sequence.

Think of the SelfAttentionLayer like a group of spotlights at a theater performance:

- Each spotlight (attention head) can focus on different actors on stage
- For each actor, the spotlights decide which other actors are most relevant to them
- The spotlights assign importance scores to these relationships
- This helps the network understand who is interacting with whom, and how

For example, in a sentence like "The cat sat on the mat because it was tired":

- Traditional networks might struggle to figure out what "it" refers to
- Self-attention can learn that "it" has a strong relationship with "cat"
- This helps the network understand that the cat was tired, not the mat

Multi-head attention (using multiple "spotlights") allows the layer to focus on different types
of relationships simultaneously, such as grammatical structure, semantic meaning, and contextual clues.

Self-attention is a cornerstone of modern natural language processing and has revolutionized
how neural networks handle sequential data like text, time series, and even images.

## How It Works

The SelfAttentionLayer implements the self-attention mechanism, a key component of transformer architectures.
It allows each position in a sequence to attend to all positions within the same sequence, enabling the model
to capture long-range dependencies and relationships. The layer uses the scaled dot-product attention mechanism
with multiple attention heads, which allows it to focus on different aspects of the input simultaneously.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfAttentionLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `SelfAttentionLayer` class with a scalar activation function. |
| `SelfAttentionLayer(Int32,Int32,Int32,IVectorActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `SelfAttentionLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the attention sparsity auxiliary loss. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (attention sparsity regularization) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BroadcastBias(Tensor<>,Int32)` | Broadcasts bias vector across the batch dimension. |
| `ComputeAuxiliaryLoss` | Initializes the layer's internal parameters based on the sequence length, embedding dimension, and head count. |
| `EnsureInitialized` | Ensures the Q/K/V/bias tensors are allocated and populated. |
| `Forward(Tensor<>)` | Performs the forward pass of the self-attention layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the attention sparsity auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetMetadata` | Returns layer-specific metadata required for cloning/serialization. |
| `GetParameterGradients` | Resets the internal state of the self-attention layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the self-attention layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weight matrices and bias vector with proper scaling. |
| `InitializeTensor(Tensor<>,)` | Initializes a 2D tensor with small random values scaled by the provided factor. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the self-attention layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the self-attention layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embeddingDimension` | The dimension of the input and output embeddings. |
| `_headCount` | The number of attention heads used in the multi-head attention mechanism. |
| `_headDimension` | The dimension of each attention head. |
| `_isInitialized` | True once `EnsureInitialized` has allocated and populated the Q/K/V weight tensors and the output bias. |
| `_keyWeights` | Tensor of weights for transforming input embeddings into key vectors. |
| `_keyWeightsGradient` | Stores the gradients of the loss with respect to the key weight parameters. |
| `_lastAttentionScores` | Stores the attention score tensor from the most recent forward pass for use in backpropagation. |
| `_lastInput` | Stores the input tensor from the most recent forward pass for use in backpropagation. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass for use in backpropagation. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outputBias` | Tensor of biases added to the output of the attention mechanism. |
| `_outputBiasGradient` | Stores the gradients of the loss with respect to the output bias parameters. |
| `_queryWeights` | Tensor of weights for transforming input embeddings into query vectors. |
| `_queryWeightsGradient` | Stores the gradients of the loss with respect to the query weight parameters. |
| `_queryWeightsHalf` | fp16-resident copies of the Q/K/V weight matrices, used only when `LowPrecisionResident` is set (foundation-scale inference). |
| `_sequenceLength` | The length of the input sequence. |
| `_valueWeights` | Tensor of weights for transforming input embeddings into value vectors. |
| `_valueWeightsGradient` | Stores the gradients of the loss with respect to the value weight parameters. |

