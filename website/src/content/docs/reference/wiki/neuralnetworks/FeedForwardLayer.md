---
title: "FeedForwardLayer<T>"
description: "Represents a fully connected (dense) feed-forward layer in a neural network."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a fully connected (dense) feed-forward layer in a neural network.

## For Beginners

A feed-forward layer is like a voting system where every input gets to vote on every output.

Imagine you have 3 inputs and 2 outputs:

- Each input has a different level of influence (weight) on each output
- Each output has its own starting value (bias)
- The layer calculates each output by combining all input influences plus the bias
- Finally, an activation function adds non-linearity (like setting a threshold)

For example:

- Input: [0.2, 0.5, 0.1] (representing features from previous layer)
- Weights: [[0.1, 0.8], [0.4, 0.3], [0.7, 0.2]] (each input's influence on each output)
- Biases: [0.1, -0.2] (starting values for each output)
- Output before activation: [0.2×0.1 + 0.5×0.4 + 0.1×0.7 + 0.1, 0.2×0.8 + 0.5×0.3 + 0.1×0.2 - 0.2]

= [0.39, 0.33]

- After activation (e.g., ReLU): [0.39, 0.33] (since both are already positive)

Feed-forward layers are the building blocks of many neural networks. Multiple
feed-forward layers stacked together form a "deep" neural network that can
learn increasingly complex patterns.

## How It Works

A feed-forward layer, also known as a fully connected or dense layer, is one of the most common
types of neural network layers. It connects every input neuron to every output neuron with
learnable weights. Each output neuron also has a learnable bias term. The layer applies a linear
transformation followed by an activation function to produce its output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeedForwardLayer(Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `FeedForwardLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Input` | The input tensor from the last forward pass, saved for backpropagation. |
| `IsInitialized` | Gets whether this layer's parameters have been allocated and initialized. |
| `Output` | The output tensor from the last forward pass, saved for backpropagation. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `PreActivationOutput` | Cached pre-activation output (after linear transform, before activation function). |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | The computation engine (CPU or GPU) for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureInitialized` | Allocates and initializes weight and bias tensors on first use. |
| `EnsureWeightShapeForInput(Int32)` | Rebuilds the weight matrix around a new input-feature width, preserving the overlapping slice of pretrained weights and Xavier-initializing the new rows. |
| `Forward(Tensor<>)` | Performs the forward pass of the feed-forward layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetBiasesTensor` | Gets the bias tensor for JIT compilation and graph composition. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeightsTensor` | Gets the weight tensor for JIT compilation and graph composition. |
| `OnFirstForward(Tensor<>)` | Resolves input feature size from input.Shape[^1] on first forward. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the weights and biases using the calculated gradients and the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The bias values for each output neuron. |
| `_biasesGradient` | The gradients for the biases, computed during backpropagation. |
| `_inputSize` | Stored input dimension for lazy initialization. |
| `_isInitialized` | Whether the weight and bias tensors have been allocated and initialized. |
| `_outputSize` | Stored output dimension for lazy initialization. |
| `_weights` | The weight matrix connecting input neurons to output neurons. |
| `_weightsGradient` | The gradients for the weights, computed during backpropagation. |

