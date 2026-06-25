---
title: "SubpixelConvolutionalLayer<T>"
description: "Represents a subpixel convolutional layer that performs convolution followed by pixel shuffling for upsampling."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a subpixel convolutional layer that performs convolution followed by pixel shuffling for upsampling.

## For Beginners

This layer helps make images larger and more detailed in neural networks.

Think of it like rearranging a small mosaic to create a larger picture:

- First, the layer creates many detailed patterns from the input (convolution step)
- Then, it rearranges these patterns to form a larger, higher-resolution output (pixel shuffling step)

For example, if you're working with a low-resolution image that's 32×32 pixels, this layer can help
transform it into a higher-resolution image of 64×64 or 128×128 pixels by intelligently filling in 
the details between the original pixels.

This is often used in applications like:

- Making blurry images clearer (super-resolution)
- Generating detailed images from rough sketches
- Converting low-quality videos to higher quality

## How It Works

A subpixel convolutional layer combines convolution with a pixel shuffling operation to efficiently increase 
spatial resolution of feature maps. It first applies convolution to produce an output with more channels, then 
rearranges these channels into a higher resolution output with fewer channels. This approach is particularly 
useful for super-resolution tasks and generative models where upsampling is required.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubpixelConvolutionalLayer(Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `SubpixelConvolutionalLayer` class with scalar activation function. |
| `SubpixelConvolutionalLayer(Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `SubpixelConvolutionalLayer` class with vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildSubpixelOptimizerState(String)` | Builds the optimizer state for a specific subpixel conv parameter. |
| `CalculateInputShape(Int32,Int32,Int32)` | Calculates the input shape for the layer based on specified dimensions. |
| `CalculateOutputShape(Int32,Int32,Int32)` | Calculates the output shape for the layer based on specified dimensions. |
| `ComputeActivationGradient(Tensor<>,Tensor<>)` | Computes the gradient with respect to the activation function. |
| `EnsureSubpixelOptimizerState(IDirectGpuBackend,GpuOptimizerType)` | Ensures GPU optimizer state buffers exist for all subpixel conv parameters. |
| `Forward(Tensor<>)` | Performs the forward pass of the subpixel convolutional layer. |
| `ForwardGpu(Tensor<>[])` | Performs the GPU-resident forward pass of the subpixel convolutional layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeWeights` | Initializes the weights and biases of the layer using Xavier initialization. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the layer and reinitializes weights. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using calculated gradients and momentum. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters using GPU-based optimizer. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Indicates whether a batch dimension was added during forward pass. |
| `_biasGradients` | The gradients of the loss with respect to the biases, computed during backward pass. |
| `_biasMomentum` | The accumulated momentum for bias updates. |
| `_biases` | The bias values added after the convolution operation. |
| `_inputDepth` | The number of channels in the input tensor. |
| `_kernelGradients` | The gradients of the loss with respect to the kernels, computed during backward pass. |
| `_kernelMomentum` | The accumulated momentum for kernel updates. |
| `_kernelSize` | The size of the convolutional kernel (filter). |
| `_kernels` | The convolutional kernels (filters) used by the layer. |
| `_lastInput` | The cached input from the last forward pass. |
| `_lastOutput` | The cached output from the last forward pass. |
| `_momentumFactor` | The factor controlling how much previous gradients influence current updates. |
| `_originalInputShape` | Stores the original input shape before any reshaping. |
| `_outputDepth` | The number of channels in the output tensor after upscaling. |
| `_upscaleFactor` | The factor by which to increase spatial dimensions (height and width). |
| `_weightDecay` | The coefficient for L2 weight regularization (weight decay). |

