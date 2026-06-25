---
title: "AttentionLayer<T>"
description: "Represents an Attention Layer for focusing on relevant parts of input sequences."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents an Attention Layer for focusing on relevant parts of input sequences.

## For Beginners

An Attention Layer helps the network focus on important parts of the input.

Think of it like reading a long document to answer a question:

- Instead of remembering every word, you focus on key sentences or phrases
- The attention mechanism does something similar for the neural network
- It helps the network decide which parts of the input are most relevant for the current task

Common applications include:

- Machine translation (focusing on relevant words when translating)
- Image captioning (focusing on relevant parts of an image when describing it)
- Speech recognition (focusing on important audio segments)

The key advantage is that it allows the network to handle long sequences more effectively
by focusing on the most relevant parts rather than trying to remember everything.

## How It Works

The Attention Layer is a mechanism that allows a neural network to focus on different parts of the input
sequence when producing each element of the output sequence. It computes a weighted sum of the input sequence,
where the weights (attention weights) are determined based on the relevance of each input element to the current output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AttentionLayer(Int32,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the AttentionLayer class with scalar activation. |
| `AttentionLayer(Int32,IVectorActivationFunction<>)` | Initializes a new instance of the AttentionLayer class with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for attention entropy regularization. |
| `InputPorts` | Declares named input ports for this multi-input layer. |
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary loss (attention entropy regularization) during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for the AttentionLayer, which is attention entropy regularization. |
| `CreateScalarTensor(,Int32[])` | Creates a tensor filled with a scalar value. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Performs the forward pass of the attention mechanism. |
| `Forward(Tensor<>[])` | Performs the forward pass of the attention mechanism with multiple inputs. |
| `ForwardCrossAttention(Tensor<>,Tensor<>,Tensor<>)` | Performs cross-attention, where query comes from one input and key/value come from another, optionally with an attention mask applied. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for the attention mechanism. |
| `ForwardMaskedAttention(Tensor<>,Tensor<>)` | Performs masked self-attention, where query, key, and value all come from the same input, but with an attention mask applied. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the attention regularization. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Retrieves the current parameters of the layer. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeTensor(Int32[],)` | Initializes a tensor with random values scaled by a given factor. |
| `OnFirstForward(Tensor<>)` | Resolves input feature size on first forward and allocates Q/K/V/O weights. |
| `ResetState` | Resets the state of the attention layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters based on the computed gradients and a learning rate. |
| `UpdateParameters(Vector<>)` | Updates the layer's parameters with the provided values. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_Wk` | The weight tensor for the key transformation. |
| `_Wo` | The weight tensor for the output projection (Wo). |
| `_Wq` | The weight tensor for the query transformation. |
| `_Wv` | The weight tensor for the value transformation. |
| `_attentionSize` | The size of the attention mechanism (typically smaller than the input size). |
| `_dWk` | Gradient of the weight tensor for the key transformation. |
| `_dWo` | Gradient of the weight tensor for the output projection. |
| `_dWq` | Gradient of the weight tensor for the query transformation. |
| `_dWv` | Gradient of the weight tensor for the value transformation. |
| `_inputSize` | The size of the input features. |
| `_inputWas2D` | Tracks whether the last input was originally 2D (and thus reshaped to 3D). |
| `_lastAttentionEntropy` | Stores the last computed attention entropy for diagnostics. |
| `_lastAttentionOutput` | Cached attention output before output projection (Wo), used for backward pass. |
| `_lastAttentionWeights` | The last attention weights computed by the layer. |
| `_lastInput` | The last input processed by the layer. |
| `_lastKeyInput` | The cached key input from the last forward pass (for cross-attention backward). |
| `_lastMask` | The cached attention mask from the last forward pass. |
| `_lastQueryInput` | The cached query input from the last forward pass (for cross-attention backward). |
| `_lastUsedMask` | Tracks whether the last forward pass used an attention mask. |
| `_lastValueInput` | The cached value input from the last forward pass. |
| `_lastWasCrossAttention` | Tracks whether the last forward pass used cross-attention (separate Q and K/V sources). |
| `_originalInputShape` | Stores the original input shape for restoring higher-rank tensor output. |

