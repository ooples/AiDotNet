---
title: "DigitCapsuleLayer<T>"
description: "Represents a digit capsule layer that implements the dynamic routing algorithm between capsules."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a digit capsule layer that implements the dynamic routing algorithm between capsules.

## For Beginners

A capsule layer is a special type of neural network layer that groups neurons together.

Think of regular neural networks as looking at individual puzzle pieces (like detecting edges or corners). 
A capsule network looks at how these pieces fit together to form objects.

For example, in image recognition:

- Regular neurons might detect a wheel, a window, and a door
- Capsules understand that these parts can make up a car, and how those parts relate to each other

This layer specifically handles digit recognition, taking information from previous capsule layers
and determining which digit is most likely present in the input.

## How It Works

A digit capsule layer extends the concept of traditional neural networks by using groups of neurons
(capsules) that encapsulate various properties of entities. This implementation is based on the 
CapsNet architecture proposed by Hinton et al., which uses a dynamic routing algorithm to determine 
how lower-level capsules should send their output to higher-level capsules.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DigitCapsuleLayer(Int32,Int32,Int32)` | Lazy constructor: resolves `inputCapsules` and `inputCapsuleDimension` from `input.Shape[^2..]` on first `Tensor{`. |
| `DigitCapsuleLayer(Int32,Int32,Int32,Int32,Int32)` | Initializes a new instance of the `DigitCapsuleLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureInitialized` |  |
| `Forward(Tensor<>)` | Performs the forward pass of the digit capsule layer using dynamic routing. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass through the digit capsule layer. |
| `GetParameterGradients` | Sets the trainable parameters of the layer from a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weight parameters with small random values. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's weights using the calculated gradients and the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputCapsuleDimension` | The dimension (number of values) of each input capsule vector. |
| `_inputCapsules` | The number of capsules in the input layer. |
| `_lastCouplings` | The coupling coefficients from the last routing iteration, saved for backpropagation. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_lastOutput` | The output tensor from the last forward pass, saved for backpropagation. |
| `_numClasses` | The number of classes (output capsules) that this layer can identify. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outputCapsuleDimension` | The dimension (number of values) of each output capsule vector. |
| `_routingIterations` | The number of iterations to use in the dynamic routing algorithm. |
| `_weights` | The weight tensor connecting input capsules to output capsules. |
| `_weightsGradient` | Gradients for the weight tensor, used during backpropagation. |

