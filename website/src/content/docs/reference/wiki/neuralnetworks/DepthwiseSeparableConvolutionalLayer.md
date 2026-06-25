---
title: "DepthwiseSeparableConvolutionalLayer<T>"
description: "Represents a depthwise separable convolutional layer that performs convolution as two separate operations."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a depthwise separable convolutional layer that performs convolution as two separate operations.

## For Beginners

A depthwise separable convolution is like a more efficient way to filter an image.

Think of it as a two-step process:

- First step (depthwise): Apply separate filters to each input channel (like filtering red, green, and blue separately)
- Second step (pointwise): Mix these filtered channels together (like combining the filtered colors)

For example, in image processing:

- Standard convolution might use 100,000 calculations for a single operation
- Depthwise separable convolution might do the same job with only 10,000 calculations

This makes your neural network faster and smaller while still capturing important patterns.
It's commonly used in mobile and edge devices where efficiency is critical.

## How It Works

A depthwise separable convolutional layer splits the standard convolution operation into two parts:
a depthwise convolution, which applies a single filter per input channel, and a pointwise convolution,
which uses 1×1 convolutions to combine the outputs. This approach dramatically reduces the number of
parameters and computational cost compared to standard convolution.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DepthwiseSeparableConvolutionalLayer(Int32,Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `DepthwiseSeparableConvolutionalLayer` class with the specified parameters and a scalar activation function. |
| `DepthwiseSeparableConvolutionalLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `DepthwiseSeparableConvolutionalLayer` class with the specified parameters and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training through backpropagation. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivation()` | Applies the activation function to a single value. |
| `ApplyActivationDerivative(,)` | Applies the derivative of the activation function during backpropagation. |
| `CalculateInputShape(Int32,Int32,Int32)` | Calculates the input shape for the layer based on input dimensions. |
| `CalculateOutputDimension(Int32,Int32,Int32,Int32)` | Calculates the output dimension after applying a convolution operation. |
| `CalculateOutputShape(Int32,Int32,Int32)` | Calculates the output shape for the layer based on output dimensions. |
| `DepthwiseConvolution(Tensor<>)` | Applies the depthwise convolution step to the input data. |
| `Forward(Tensor<>)` | Processes the input data through the depthwise separable convolutional layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass using fused DepthwiseConv2D + pointwise Conv2D + Bias + Activation. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the kernel weights and biases with appropriate random values. |
| `OnFirstForward(Tensor<>)` | Resolves input depth and output spatial dims on first forward. |
| `PointwiseConvolution(Tensor<>)` | Applies the pointwise convolution step to the depthwise convolution output. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters (kernel weights and biases) using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters on GPU using the configured optimizer. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Indicates whether a batch dimension was added during forward pass. |
| `_biases` | The bias values added to each output channel. |
| `_biasesGradient` | Calculated gradients for the biases during backpropagation. |
| `_depthwiseKernels` | The filter kernels used for the depthwise convolution step. |
| `_depthwiseKernelsGradient` | Calculated gradients for the depthwise kernels during backpropagation. |
| `_inputDepth` | The number of channels in the input data. |
| `_kernelSize` | The size of each depthwise filter kernel. |
| `_lastDepthwiseOutput` | Stored output from the depthwise convolution step, used for backpropagation. |
| `_lastInput` | Stored input data from the most recent forward pass, used for backpropagation. |
| `_lastOutput` | Stored final output data from the most recent forward pass, used for backpropagation. |
| `_lastPreActivation` | Stored pre-activation output from the most recent forward pass, used for backpropagation. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outputDepth` | The number of channels in the output data. |
| `_padding` | The amount of zero-padding added to the input data before convolution. |
| `_pointwiseKernels` | The filter kernels used for the pointwise convolution step. |
| `_pointwiseKernelsGradient` | Calculated gradients for the pointwise kernels during backpropagation. |
| `_stride` | The step size for moving the kernel across the input data. |

