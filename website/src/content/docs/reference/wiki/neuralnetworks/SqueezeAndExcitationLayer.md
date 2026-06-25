---
title: "SqueezeAndExcitationLayer<T>"
description: "Represents a Squeeze-and-Excitation layer that recalibrates channel-wise feature responses adaptively."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Squeeze-and-Excitation layer that recalibrates channel-wise feature responses adaptively.

## For Beginners

This layer helps the neural network focus on the most important features.

Think of it like how your brain works when looking at a picture:

- First, you get a rough idea of what's in the image (the "squeeze" step)
- Then, you decide which parts to pay more attention to (the "excitation" step)
- Finally, you look at the image again with this focused attention

For example, if the network is processing an image of a cat, the Squeeze-and-Excitation layer might:

- First compress all the information to understand "this is probably a cat"
- Then decide to pay more attention to features that look like ears, whiskers, and fur
- Finally enhance those important features in the original image data

This helps the network become more accurate and efficient by focusing on what matters most.

## How It Works

A Squeeze-and-Excitation layer enhances the representational power of a network by explicitly modeling the 
interdependencies between channels. It does this by performing two operations:

1. "Squeeze" - aggregating feature maps across spatial dimensions to produce a channel descriptor
2. "Excitation" - using this descriptor to recalibrate the original feature maps channel-wise

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SqueezeAndExcitationLayer(Int32,Int32,IActivationFunction<>,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `SqueezeAndExcitationLayer` class with scalar activation functions. |
| `SqueezeAndExcitationLayer(Int32,Int32,IVectorActivationFunction<>,IVectorActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `SqueezeAndExcitationLayer` class with vector activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SparsityWeight` | Gets or sets the weight for L1 sparsity regularization on attention weights. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyTensorActivation(Tensor<>,Boolean)` | Applies the appropriate activation function to the input tensor. |
| `ApplyTensorActivationDerivative(Tensor<>,Boolean)` | Applies the derivative of the activation function for backpropagation. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer based on channel attention regularization. |
| `Forward(Tensor<>)` | Performs the forward pass of the Squeeze-and-Excitation layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass of the Squeeze-and-Excitation layer on GPU tensors. |
| `ForwardGpu1D(Tensor<>,Tensor<>,Tensor<>,IDirectGpuBackend,DirectGpuTensorEngine)` | GPU forward pass for 1D inputs [C] - reshape to [1, C] and process. |
| `ForwardGpu2D(Tensor<>,Tensor<>,Tensor<>,IDirectGpuBackend,DirectGpuTensorEngine)` | GPU forward pass for 2D inputs [B, C] - no squeeze needed. |
| `ForwardGpu3D(Tensor<>,Tensor<>,Tensor<>,IDirectGpuBackend,DirectGpuTensorEngine)` | GPU forward pass for 3D inputs [B, L, C] - uses MeanAxis for squeeze. |
| `ForwardGpu4D(Tensor<>,Tensor<>,Tensor<>,IDirectGpuBackend,DirectGpuTensorEngine)` | GPU forward pass for 4D inputs [B, H, W, C] - uses GlobalAvgPool2D. |
| `ForwardGpuND(Tensor<>,Tensor<>,Tensor<>,IDirectGpuBackend,DirectGpuTensorEngine)` | GPU forward pass for higher-rank tensors [B, D1, D2, ..., C] - flatten spatial dims. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetFirstActivationType` | Gets the FusedActivationType corresponding to the first activation function. |
| `GetParameterGradients` | Sets the trainable parameters of the layer from a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetSecondActivationType` | Gets the FusedActivationType corresponding to the second activation function. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeTensor1D(Tensor<>,)` | Initializes a 1D tensor with small random values scaled by the specified factor. |
| `InitializeTensor2D(Tensor<>,)` | Initializes a 2D tensor with small random values scaled by the specified factor. |
| `InitializeWeights` | Initializes the weights and biases of the layer with small random values. |
| `ResetState` | Resets the internal state of the Squeeze-and-Excitation layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters using the calculated gradients and the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias1` | The bias values for the first fully connected layer. |
| `_bias1Gradient` | The gradient of the loss with respect to _bias1 from the most recent backward pass. |
| `_bias2` | The bias values for the second fully connected layer. |
| `_bias2Gradient` | The gradient of the loss with respect to _bias2 from the most recent backward pass. |
| `_channels` | The number of input and output channels in the layer. |
| `_firstActivation` | The activation function applied after the first fully connected layer. |
| `_firstVectorActivation` | The vector activation function applied after the first fully connected layer. |
| `_lastChannelAttentionLoss` | Stores the last computed channel attention regularization loss for diagnostic purposes. |
| `_lastExcitationWeights` | Caches the excitation weights from the forward pass for auxiliary loss computation. |
| `_lastInput` | The input tensor from the most recent forward pass. |
| `_lastOutput` | The output tensor from the most recent forward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_reducedChannels` | The number of channels in the bottleneck (reduced dimension). |
| `_secondActivation` | The activation function applied after the second fully connected layer. |
| `_secondVectorActivation` | The vector activation function applied after the second fully connected layer. |
| `_weights1` | The weights for the first fully connected layer. |
| `_weights1Gradient` | The gradient of the loss with respect to _weights1 from the most recent backward pass. |
| `_weights2` | The weights for the second fully connected layer. |
| `_weights2Gradient` | The gradient of the loss with respect to _weights2 from the most recent backward pass. |

