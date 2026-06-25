---
title: "FullyConnectedLayer<T>"
description: "Represents a fully connected layer in a neural network where every input neuron connects to every output neuron."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a fully connected layer in a neural network where every input neuron connects to every output neuron.

## For Beginners

A fully connected layer connects every input to every output, like a complete web of connections.

Imagine you have inputs representing different features:

- Each feature (input) connects to every possible output
- Each connection has a strength (weight) that can be adjusted
- Each output also has a starting value (bias)

For example, in an image classification task:

- Inputs might be flattened features from convolutional layers
- Each output might represent a score for a different category
- The connections (weights) learn which features are important for each category

Fully connected layers are excellent at combining features to make final decisions.
They're often used toward the end of a neural network to interpret the features
extracted by earlier layers.

## How It Works

A fully connected layer, also known as a dense layer, is a fundamental building block in neural networks.
It connects every input neuron to every output neuron with learnable weights. Each output neuron also has
a learnable bias term. The layer applies a linear transformation followed by an activation function to
produce its output. Fully connected layers are particularly useful for learning complex patterns and 
for classification tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FullyConnectedLayer(Int32,Int32,IActivationFunction<>)` | Eager constructor that allocates and initializes the weight/bias tensors immediately for a known input size — the PyTorch `nn.Linear(in_features, out_features)` convention. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters (weights + biases). |
| `SupportsGpuExecution` | Gets whether this layer has a GPU implementation. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Clears stored gradients for weights and biases. |
| `Forward(Tensor<>)` | Performs the forward pass of the fully connected layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass, keeping tensors on GPU. |
| `GetBiases` |  |
| `GetParameterGradients` | Gets the gradients of all trainable parameters in this layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` |  |
| `InitializeParameters` | Initializes the weights and biases with appropriate values for effective training. |
| `OnFirstForward(Tensor<>)` | Resolves input feature size from input.Shape[^1] on first forward and allocates weights. |
| `ResetState` | Resets the internal state of the layer. |
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
| `_inputWas1D` | Tracks whether the last forward pass input was rank-1, so backward can preserve rank. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_lastOutput` | The output tensor from the last forward pass, saved for backpropagation. |
| `_weights` | The weight matrix connecting input neurons to output neurons. |
| `_weightsGradient` | The gradients for the weights, computed during backpropagation. |

