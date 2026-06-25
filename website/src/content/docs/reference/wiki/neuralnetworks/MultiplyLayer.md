---
title: "MultiplyLayer<T>"
description: "Represents a layer that performs element-wise multiplication of multiple input tensors."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a layer that performs element-wise multiplication of multiple input tensors.

## For Beginners

This layer multiplies tensors together, element by element.

Think of it like multiplying numbers together in corresponding positions:

- If you have two vectors [1, 2, 3] and [4, 5, 6]
- The result would be [1×4, 2×5, 3×6] = [4, 10, 18]

This is useful for:

- Controlling information flow (like gates in LSTM or GRU cells)
- Applying masks (to selectively focus on certain values)
- Combining features in a multiplicative way

For example, in an attention mechanism, you might multiply feature values by attention weights
to focus on important features and diminish the influence of less relevant ones.

## How It Works

The MultiplyLayer performs element-wise multiplication (Hadamard product) of two or more input tensors
of identical shape. This operation can be useful for implementing gating mechanisms, attention masks,
or feature-wise interactions in neural networks. The layer requires that all input tensors have the
same shape, and it produces an output tensor of that same shape.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiplyLayer(Int32[][],IActivationFunction<>)` | Initializes a new instance of the `MultiplyLayer` class with the specified input shapes and a scalar activation function. |
| `MultiplyLayer(Int32[][],IVectorActivationFunction<>)` | Initializes a new instance of the `MultiplyLayer` class with the specified input shapes and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` |  |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BackwardGpu(Tensor<>)` | Computes the gradients of the loss with respect to the inputs on the GPU. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Named multi-input forward pass. |
| `Forward(Tensor<>[])` | Performs the forward pass of the multiply layer with multiple input tensors. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using actual GPU element-wise multiplication. |
| `GetParameters` | Gets all trainable parameters from the multiply layer as a single vector. |
| `ResetState` | Resets the internal state of the multiply layer. |
| `UpdateParameters()` | Updates the parameters of the multiply layer using the calculated gradients. |
| `ValidateInputShapes(Int32[][])` | Validates that the input shapes are appropriate for a multiply layer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputPortsCache` | Declares named input ports for this multi-input layer. |
| `_lastInputs` | The input tensors from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |

