---
title: "MemoryWriteLayer<T>"
description: "Represents a layer that writes to a memory tensor using an attention mechanism."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that writes to a memory tensor using an attention mechanism.

## For Beginners

This layer helps a neural network store information in memory.

Think of it like deciding what to write in a notebook:

- You have some new information (your current input)
- You have a notebook with existing notes (your memory)
- The layer decides which pages of the notebook are relevant to your new information
- It then writes the new information on those pages, focusing more on the most relevant ones

For example, if your input represents new information about "France has a beautiful capital city",
the layer would focus on memory locations related to France and update them with this new information.

This is similar to how we humans selectively update our memories with new information, rather than
storing everything in completely new locations.

## How It Works

The MemoryWriteLayer implements a form of attention-based memory writing. It computes attention scores
between the input and memory tensors, using these scores to determine where to write new information.
This approach allows the layer to selectively update memory based on the current input. The layer uses
a query-key-value attention mechanism where queries and keys determine where to write, and values determine
what to write.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MemoryWriteLayer(Int32,IActivationFunction<>)` | Initializes a new instance of the `MemoryWriteLayer` class with the specified dimensions and a scalar activation function. |
| `MemoryWriteLayer(Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `MemoryWriteLayer` class with the specified dimensions and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BroadcastBiases(Tensor<>,Int32)` | Broadcasts a bias tensor across the batch dimension. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer based on attention sparsity regularization. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>)` | Performs the forward pass of the memory write layer with just the input tensor. |
| `Forward(Tensor<>,Tensor<>)` | Performs the forward pass of the memory write layer with input and memory tensors. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Resets the internal state of the memory write layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters from the memory write layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the layer's weights and biases. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with random values scaled by the given factor. |
| `OnFirstForward(Tensor<>)` | Resolves input feature size from input.Shape on first forward and allocates Q/K/V weights. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters for the memory write layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the memory write layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputPortsCache` | Declares named input ports for this multi-input layer. |
| `_keyWeights` | The weight tensor used to transform the input into key vectors. |
| `_keyWeightsGradient` | The gradient of the loss with respect to the key weights. |
| `_lastAttentionScores` | The attention scores tensor from the most recent forward pass. |
| `_lastAttentionSparsityLoss` | Stores the last computed attention sparsity loss for diagnostic purposes. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_lastMemory` | The memory tensor from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |
| `_lastValues` | The transformed values tensor from the most recent forward pass. |
| `_lastWriteValues` | The write values tensor from the most recent forward pass. |
| `_outputBias` | The bias tensor added to the output. |
| `_outputBiasGradient` | The gradient of the loss with respect to the output bias. |
| `_outputWeights` | The weight tensor applied to the output after value transformation. |
| `_outputWeightsGradient` | The gradient of the loss with respect to the output weights. |
| `_queryWeights` | The weight tensor used to transform the input into query vectors. |
| `_queryWeightsGradient` | The gradient of the loss with respect to the query weights. |
| `_valueWeights` | The weight tensor used to transform the input into value vectors. |
| `_valueWeightsGradient` | The gradient of the loss with respect to the value weights. |

