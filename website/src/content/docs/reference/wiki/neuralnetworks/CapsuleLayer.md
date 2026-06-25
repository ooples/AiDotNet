---
title: "CapsuleLayer<T>"
description: "Represents a capsule neural network layer that encapsulates groups of neurons to better preserve spatial information."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a capsule neural network layer that encapsulates groups of neurons to better preserve spatial information.

## For Beginners

A capsule layer is an advanced type of neural network layer that works differently 
from standard layers.

Traditional neural network layers use single numbers to represent features, but capsule layers use groups of numbers
(vectors) that can capture more information:

- Each "capsule" is a group of neurons that work together
- The length of a capsule's output tells you how likely something exists
- The direction of a capsule's output tells you about its properties (like position, size, rotation)

For example, if detecting faces in images:

- A traditional network might have neurons that detect eyes, nose, mouth separately
- A capsule network would understand how these parts relate to each other spatially

This helps the network recognize objects even when they're viewed from different angles or positions,
which is something traditional networks struggle with.

## How It Works

A capsule layer is a specialized neural network layer that groups neurons into "capsules," where each capsule 
represents a specific entity or feature. Unlike traditional neural networks that use scalar outputs, capsules 
output vectors whose length represents the probability of the entity's existence and whose orientation encodes 
the entity's properties. Capsule layers use dynamic routing between capsules, which helps preserve hierarchical 
relationships between features and improves the network's ability to recognize objects from different viewpoints.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CapsuleLayer(Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `CapsuleLayer` class with specified dimensions and routing iterations. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the routing entropy auxiliary loss. |
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (routing entropy regularization) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Tensor<>)` | Applies the softmax activation function to the input tensor. |
| `ClearGradients` | Resets the internal state of the capsule layer. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for routing entropy regularization. |
| `Forward(Tensor<>)` | Performs the forward pass of the capsule layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass through the capsule layer. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the routing entropy auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Sets the trainable parameters for the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters from the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the layer's transformation matrix and bias parameters. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `OnFirstForward(Tensor<>)` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

