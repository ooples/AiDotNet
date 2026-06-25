---
title: "SeparableConvolutionalLayer<T>"
description: "Represents a separable convolutional layer that decomposes standard convolution into depthwise and pointwise operations."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a separable convolutional layer that decomposes standard convolution into depthwise and pointwise operations.

## For Beginners

This layer processes images or other grid-like data more efficiently than standard convolution.

Think of it like a two-step process:

- First step (depthwise): Applies filters to each input channel separately to extract features
- Second step (pointwise): Combines these features across all channels to create new feature maps

Benefits include:

- Fewer calculations needed (faster processing)
- Fewer parameters to learn (uses less memory)
- Often similar performance to standard convolution

For example, in image processing, the depthwise convolution might detect edges in each color channel separately,
while the pointwise convolution would combine these edges into more complex features like shapes or textures.

## How It Works

A separable convolutional layer splits the standard convolution operation into two simpler operations: 
a depthwise convolution followed by a pointwise convolution. This factorization significantly reduces 
computational complexity and number of parameters while maintaining similar model expressiveness.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeparableConvolutionalLayer(Int32,Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `SeparableConvolutionalLayer` class with a scalar activation function. |
| `SeparableConvolutionalLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `SeparableConvolutionalLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training through backpropagation. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU-accelerated execution. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateOutputShape(Int32[],Int32,Int32,Int32,Int32)` | Calculates the output shape of the separable convolutional layer based on input shape and parameters. |
| `ConvertDepthwiseKernelFromNCHW(Tensor<>)` | Converts depthwise kernel from [inputDepth, 1, kernelSize, kernelSize] back to [inputDepth, kernelSize, kernelSize, 1] format. |
| `ConvertDepthwiseKernelToNCHW(Tensor<>)` | Converts depthwise kernel from [inputDepth, kernelSize, kernelSize, 1] to [inputDepth, 1, kernelSize, kernelSize] format. |
| `ConvertPointwiseKernelFromNCHW(Tensor<>)` | Converts pointwise kernel from [outputDepth, inputDepth, 1, 1] back to [inputDepth, 1, 1, outputDepth] format. |
| `ConvertPointwiseKernelToNCHW(Tensor<>)` | Converts pointwise kernel from [inputDepth, 1, 1, outputDepth] to [outputDepth, inputDepth, 1, 1] format. |
| `Forward(Tensor<>)` | Performs the forward pass of the separable convolutional layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU, keeping all tensors GPU-resident. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the layer's parameters (kernels and biases). |
| `InitializeTensor(Tensor<>,)` | Initializes a tensor with scaled random values. |
| `OnFirstForward(Tensor<>)` | Resolves input depth from input.Shape (NHWC last axis) and output spatial dims on first forward. |
| `ResetState` | Resets the internal state of the separable convolutional layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients and momentum. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters on GPU using the configured optimizer. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | Bias values added to each output channel. |
| `_biasesGradient` | Gradient of the loss with respect to the biases. |
| `_biasesVelocity` | Stores the velocity for momentum-based updates of biases. |
| `_depthwiseKernels` | Kernels for the depthwise convolution operation. |
| `_depthwiseKernelsGradient` | Gradient of the loss with respect to the depthwise kernels. |
| `_depthwiseKernelsVelocity` | Stores the velocity for momentum-based updates of depthwise kernels. |
| `_inputDepth` | Number of channels in the input tensor. |
| `_kernelSize` | Size of the convolution kernel (assumed to be square). |
| `_lastInput` | Stores the input tensor from the most recent forward pass. |
| `_lastOutput` | Stores the output tensor from the most recent forward pass. |
| `_outputDepth` | Number of channels in the output tensor. |
| `_padding` | Padding applied to the input before convolution. |
| `_pointwiseKernels` | Kernels for the pointwise convolution operation. |
| `_pointwiseKernelsGradient` | Gradient of the loss with respect to the pointwise kernels. |
| `_pointwiseKernelsVelocity` | Stores the velocity for momentum-based updates of pointwise kernels. |
| `_stride` | Stride of the convolution operation. |

