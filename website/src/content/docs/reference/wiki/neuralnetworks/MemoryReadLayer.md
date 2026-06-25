---
title: "MemoryReadLayer<T>"
description: "Represents a layer that reads from a memory tensor using an attention mechanism."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that reads from a memory tensor using an attention mechanism.

## For Beginners

This layer helps a neural network retrieve information from memory.

Think of it like searching for relevant information in a book:

- You have a query (your current input)
- You have a memory (like pages of a book)
- The layer finds which parts of the memory are most relevant to your query
- It then combines those relevant parts to produce an output

For example, if your input represents a question like "What's the capital of France?",
the layer would look through memory to find information about France, give more attention
to content about its capital, and then combine this information to produce the answer "Paris".

This is similar to how modern language models can retrieve and use stored information
when answering questions.

## How It Works

The MemoryReadLayer implements a form of attention-based memory access. It computes attention scores
between the input and memory tensors, using these scores to create a weighted sum of memory values.
This approach allows the layer to selectively retrieve information from memory based on the current input.
The layer consists of key weights (for attention computation), value weights (for transforming memory values),
and output weights (for final processing).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryReadLayer(Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `MemoryReadLayer` class with the specified dimensions and a scalar activation function. |
| `MemoryReadLayer(Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `MemoryReadLayer` class with the specified dimensions and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `InputPorts` | Declares named input ports for this multi-input layer. |
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BroadcastBiases(Tensor<>,Int32)` | Broadcasts a bias tensor across the batch dimension. |
| `CombineGradients(Tensor<>,Tensor<>)` | Combines gradients for input and memory into a single tensor. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer based on attention sparsity regularization. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Performs a forward pass using a default identity-like memory tensor. |
| `Forward(Tensor<>,Tensor<>)` | Performs the forward pass of the memory read layer with input and memory tensors. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Resets the internal state of the memory read layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters from the memory read layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the layer's weights and biases. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with random values scaled by the given factor. |
| `OnFirstForward(Tensor<>)` | Resolves input feature size from input.Shape on first forward and allocates key weights. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters for the memory read layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the memory read layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_keyWeights` | The weight tensor used to transform the input into query keys. |
| `_keyWeightsGradient` | The gradient of the loss with respect to the key weights. |
| `_lastAttentionScores` | The attention scores tensor from the most recent forward pass. |
| `_lastAttentionSparsityLoss` | Stores the last computed attention sparsity loss for diagnostic purposes. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_lastMemory` | The memory tensor from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |
| `_lastTransformed` | The transformed tensor from the most recent forward pass (input to output weights). |
| `_outputBias` | The bias tensor added to the output. |
| `_outputBiasGradient` | The gradient of the loss with respect to the output bias. |
| `_outputWeights` | The weight tensor applied to the output after value transformation. |
| `_outputWeightsGradient` | The gradient of the loss with respect to the output weights. |
| `_valueWeights` | The weight tensor used to transform the memory values after attention. |
| `_valueWeightsGradient` | The gradient of the loss with respect to the value weights. |

