---
title: "GatedLinearUnitLayer<T>"
description: "Represents a Gated Linear Unit (GLU) layer in a neural network that combines linear transformation with multiplicative gating."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Gated Linear Unit (GLU) layer in a neural network that combines linear transformation with multiplicative gating.

## For Beginners

A Gated Linear Unit is like a smart filter that controls how much information flows through.

Imagine water flowing through a pipe with an adjustable valve:

- The water is the input data
- One part of the layer (linear part) processes the water
- Another part (gate) controls how much processed water flows through
- Together they decide "what information is important to keep"

For example, in language processing:

- The linear transformation might extract features from words
- The gate might decide which features are relevant to the current context
- Their combination helps the network focus on important information

GLUs are particularly good at:

- Controlling information flow through the network
- Helping gradients flow during training (preventing vanishing gradients)
- Allowing the network to selectively use information

This selectivity is valuable in many tasks, especially those involving sequences
like text or time-series data.

## How It Works

A Gated Linear Unit (GLU) is a neural network layer that combines linear transformations with a gating mechanism.
It applies two parallel linear transformations to the input: one produces a linear output, and the other produces
a gate that controls how much of the linear output passes through. The final output is the element-wise product
of the linear output and the activated gate. GLUs were introduced to help with vanishing gradient problems in
deep networks and have been particularly effective in natural language processing and sequence modeling tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GatedLinearUnitLayer(Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `GatedLinearUnitLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the GLU layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using FusedLinearGpu for efficient computation. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases with appropriate values for effective training. |
| `OnFirstForward(Tensor<>)` | Resolves input dimension on first forward and allocates linear/gate weight matrices. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the weights and biases for both paths using the calculated gradients and the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gateBias` | The bias tensor for the gating path. |
| `_gateBiasGradient` | The gradients for the gate biases, computed during backpropagation. |
| `_gateWeights` | The weight tensor for the gating path. |
| `_gateWeightsGradient` | The gradients for the gate weights, computed during backpropagation. |
| `_lastGateOutput` | The output tensor from the gating path of the last forward pass, saved for backpropagation. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_lastLinearOutput` | The output tensor from the linear path of the last forward pass, saved for backpropagation. |
| `_linearBias` | The bias tensor for the linear transformation path. |
| `_linearBiasGradient` | The gradients for the linear biases, computed during backpropagation. |
| `_linearWeights` | The weight tensor for the linear transformation path. |
| `_linearWeightsGradient` | The gradients for the linear weights, computed during backpropagation. |

