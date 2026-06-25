---
title: "OctonionLinearLayer<T>"
description: "Represents a fully connected layer using octonion-valued weights and inputs."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a fully connected layer using octonion-valued weights and inputs.

## For Beginners

This layer is like a regular dense layer, but it uses
8-dimensional numbers (octonions) instead of regular numbers.

Benefits of octonion layers:

- Can model more complex relationships with fewer parameters
- Useful for certain types of image and signal processing
- Better at capturing rotational relationships in data

The tradeoff is that computations are more expensive per parameter.

## How It Works

An octonion linear layer performs matrix-vector multiplication in the 8-dimensional
octonion algebra. Each weight and input is an octonion (8 real components), enabling
the layer to capture more complex relationships than real-valued layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OctonionLinearLayer(Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the OctonionLinearLayer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputFeatures` | Gets the number of input features (octonion-valued). |
| `OutputFeatures` | Gets the number of output features (octonion-valued). |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Resets the internal state of the layer. |
| `Forward(Tensor<>)` | Performs the forward pass through the layer. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The octonion bias tensor with shape [OutputFeatures, 8]. |
| `_biasesGradient` | Gradient for biases. |
| `_lastInput` | Stored input from forward pass for backpropagation. |
| `_lastOutput` | Stored pre-activation output for gradient computation. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_weights` | The octonion weight tensor with shape [OutputFeatures, InputFeatures, 8]. |
| `_weightsGradient` | Gradient for weights. |

