---
title: "SpatialTransformerLayer<T>"
description: "Represents a spatial transformer layer that enables spatial manipulations of data via a learnable transformation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a spatial transformer layer that enables spatial manipulations of data via a learnable transformation.

## For Beginners

This layer helps a neural network focus on the important parts of an image by learning to transform it.

Think of it like having a smart camera that can:

- Zoom in on the important objects
- Rotate images to make them easier to recognize
- Crop out distractions
- Fix distortions or perspective problems

Benefits include:

- Automatic learning of the best transformation for the task
- Improved recognition of objects regardless of their position or orientation
- Better handling of distorted or warped inputs

For example, when recognizing handwritten digits, a spatial transformer might learn to straighten
tilted digits or zoom in on the digit, making it easier for the rest of the network to classify.

## How It Works

A spatial transformer layer performs learnable geometric transformations on input feature maps. It consists of
three main components: a localization network that predicts transformation parameters, a grid generator that
creates a sampling grid, and a sampler that applies the transformation using bilinear interpolation. This allows
the network to automatically learn invariance to translation, scale, rotation, and more general warping.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpatialTransformerLayer(Int32,Int32,IActivationFunction<>,SpatialTransformerDataFormat)` | Initializes a new instance of the `SpatialTransformerLayer` class with a scalar activation function. |
| `SpatialTransformerLayer(Int32,Int32,IVectorActivationFunction<>,SpatialTransformerDataFormat)` | Initializes a new instance of the `SpatialTransformerLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `ParameterCount` | Gets a value indicating whether this layer supports training through backpropagation. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivationGradientTensor(Tensor<>,Tensor<>)` | Applies the activation function gradient to a tensor. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer based on transformation regularization. |
| `ConvertToTransformationMatrix(Tensor<>)` | Converts the transformation parameters to a 2x3 transformation matrix. |
| `Forward(Tensor<>)` | Performs the forward pass of the spatial transformer layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU tensors by applying spatial transformation. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Sets the trainable parameters of the layer from a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the localization network. |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `OnFirstForward(Tensor<>)` | Resolves input H/W on first forward and allocates the localization network weights. |
| `ResetState` | Resets the internal state of the spatial transformer layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputHadChannel` | Stores whether the original input included a channel dimension. |
| `_inputHeight` | The height of the input feature map. |
| `_inputWidth` | The width of the input feature map. |
| `_lastInput` | Stores the input tensor from the most recent forward pass. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass. |
| `_lastTransformationLoss` | Stores the last computed transformation regularization loss for diagnostic purposes. |
| `_lastTransformationMatrix` | Stores the transformation matrix from the most recent forward pass. |
| `_localizationBias1` | Biases for the first layer of the localization network. |
| `_localizationBias1Gradient` | Gradient of the loss with respect to the biases of the first localization layer. |
| `_localizationBias2` | Biases for the second layer of the localization network. |
| `_localizationBias2Gradient` | Gradient of the loss with respect to the biases of the second localization layer. |
| `_localizationWeights1` | Weights for the first layer of the localization network. |
| `_localizationWeights1Gradient` | Gradient of the loss with respect to the weights of the first localization layer. |
| `_localizationWeights2` | Weights for the second layer of the localization network. |
| `_localizationWeights2Gradient` | Gradient of the loss with respect to the weights of the second localization layer. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outputHeight` | The height of the output feature map. |
| `_outputWidth` | The width of the output feature map. |

