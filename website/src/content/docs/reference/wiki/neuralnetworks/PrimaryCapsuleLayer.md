---
title: "PrimaryCapsuleLayer<T>"
description: "Represents a primary capsule layer for capsule networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a primary capsule layer for capsule networks.

## For Beginners

This layer is the first step in creating a capsule network.

In traditional neural networks, each neuron outputs a single number indicating the presence of a feature.
In capsule networks, neurons are grouped into "capsules" where each capsule outputs a vector:

- The length of the vector represents the presence of an entity
- The orientation of the vector represents properties of that entity

Think of it like this:

- Standard neurons: "I see a nose with 90% confidence"
- Capsule neurons: "I see a nose with 90% confidence, and it's pointing 30° to the left, 

it's 2cm long, it has a slightly curved shape..."

The primary capsule layer converts traditional feature maps (from convolutional layers)
into these vector-based capsules that can capture more detailed information about the entities detected.

This approach helps the network understand spatial relationships and maintain information
about pose, orientation, and other properties that are typically lost in traditional networks.

## How It Works

The PrimaryCapsuleLayer is the first layer in a capsule network that transforms traditional scalar feature maps
into capsule vectors. It performs a convolution operation followed by reshaping the output into capsules.
Each capsule represents a group of neurons that encodes both the presence and properties of a particular entity.
This layer serves as a bridge between standard convolutional layers and higher-level capsule layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrimaryCapsuleLayer(Int32,Int32,Int32,Int32,IActivationFunction<>)` | Lazy constructor (scalar activation): resolves `inputChannels` from `input.Shape[1]` (NCHW) on first `Tensor{`. |
| `PrimaryCapsuleLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Lazy constructor (vector activation): resolves `inputChannels` from `input.Shape[1]` (NCHW) on first `Tensor{`. |
| `PrimaryCapsuleLayer(Int32,Int32,Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `PrimaryCapsuleLayer` class with the specified parameters and a scalar activation function. |
| `PrimaryCapsuleLayer(Int32,Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `PrimaryCapsuleLayer` class with the specified parameters and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureInitialized` |  |
| `ExtractPatch(Tensor<>,Int32,Int32,Int32)` | Extracts a patch from the input tensor for convolution. |
| `Forward(Tensor<>)` | Performs the forward pass of the primary capsule layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass through the primary capsule layer. |
| `GetParameterGradients` | Sets the trainable parameters for the primary capsule layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters from the primary capsule layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the layer's weights and biases. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the primary capsule layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the primary capsule layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_capsuleChannels` | The number of capsule channels. |
| `_capsuleDimension` | The dimension of each capsule. |
| `_convBias` | The bias tensor for convolution operations. |
| `_convBiasGradient` | The gradient of the loss with respect to the convolution bias. |
| `_convWeights` | The weight tensor for convolution operations. |
| `_convWeightsGradient` | The gradient of the loss with respect to the convolution weights. |
| `_inputChannels` | The number of input channels. |
| `_inputWasNCHW` | Stores whether the original input was provided in NCHW layout. |
| `_kernelSize` | The size of the convolutional kernel. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_stride` | The stride of the convolution operation. |

