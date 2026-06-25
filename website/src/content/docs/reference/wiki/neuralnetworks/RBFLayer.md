---
title: "RBFLayer<T>"
description: "Represents a Radial Basis Function (RBF) layer for neural networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Radial Basis Function (RBF) layer for neural networks.

## For Beginners

This layer works like a collection of specialized detectors.

Think of each neuron in this layer as a spotlight:

- Each spotlight has a specific location (center) in the input space
- Each spotlight has a certain brightness range (width)
- When input comes in, spotlights that are close to that input light up brightly
- Spotlights far from the input barely light up at all

For example, if you're recognizing handwritten digits:

- One spotlight might be positioned to detect curved lines (like in "8")
- Another might detect vertical lines (like in "1")
- When a "3" comes in, the spotlights for curves light up strongly, while others stay dim

This layer is particularly good at classification problems and function approximation
where the relationship between inputs and outputs is complex or non-linear.

## How It Works

The RBF layer implements a type of artificial neural network that uses radial basis functions as 
activation functions. Each neuron in this layer has a center point in the input space and responds
most strongly to inputs near that center. The response decreases as the distance from the center
increases, controlled by the width parameter of each neuron.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RBFLayer(Int32,IRadialBasisFunction<>,IInitializationStrategy<>)` | Lazy constructor: resolves `inputSize` from `input.Shape[^1]` on first `Tensor{`. |
| `RBFLayer(Int32,Int32,IRadialBasisFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `RBFLayer` class with specified dimensions and radial basis function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Resets the internal state of the RBF layer. |
| `ComputeEpsilonsFromWidths` | Converts width parameters to epsilon values for RBF kernel. |
| `ConvertEpsilonGradientsToWidthGradients(Tensor<>)` | Converts epsilon gradients to width gradients using chain rule. |
| `EnsureInitialized` |  |
| `Forward(Tensor<>)` | Performs the forward pass of the RBF layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetMetadata` |  |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the RBF layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the centers and widths of the RBF layer with random values. |
| `OnFirstForward(Tensor<>)` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the RBF layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the RBF layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_centers` | Tensor storing the center positions of each RBF neuron in the input space. |
| `_centersGradient` | Stores the gradients of the loss with respect to the center parameters. |
| `_inputSize` | Number of input features. |
| `_isInitialized` | Lazy-init flag. |
| `_lastInput` | Stores the input tensor from the most recent forward pass for use in backpropagation. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass for use in backpropagation. |
| `_numCenters` | Number of RBF neurons (output size). |
| `_rbf` | The radial basis function implementation used to compute neuron activations. |
| `_widths` | Tensor storing the width parameters for each RBF neuron. |
| `_widthsGradient` | Stores the gradients of the loss with respect to the width parameters. |

