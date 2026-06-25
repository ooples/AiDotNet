---
title: "DeconvolutionalLayer<T>"
description: "Represents a deconvolutional layer (also known as transposed convolution) in a neural network."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a deconvolutional layer (also known as transposed convolution) in a neural network.

## For Beginners

A deconvolutional layer is like zooming in on an image in a smart way.

Think of it like the reverse of a convolutional layer:

- A convolutional layer summarizes information (making images smaller)
- A deconvolutional layer expands information (making images larger)

For example, if you have a small feature map representing "cat features," a deconvolutional layer
could expand it back to a cat-shaped image.

This is particularly useful for:

- Generating images from small encoded representations
- Increasing the resolution of feature maps
- Creating detailed outputs from simplified inputs

Applications include image generation, super-resolution, and segmentation tasks where
you need to expand the spatial dimensions of your data.

## How It Works

A deconvolutional layer performs the opposite operation of a convolutional layer. While convolution
reduces spatial dimensions by applying filters, deconvolution expands spatial dimensions by applying
learnable filters to upsample the input. This is particularly useful in generative models and
image segmentation networks where upsampling is required.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeconvolutionalLayer(Int32,Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `DeconvolutionalLayer` class with the specified parameters and a scalar activation function. |
| `DeconvolutionalLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `DeconvolutionalLayer` class with the specified parameters and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDepth` | Gets the depth (number of channels) of the input data. |
| `KernelSize` | Gets the size of each filter (kernel) used in the deconvolution operation. |
| `OutputDepth` | Gets the depth (number of channels) of the output data. |
| `Padding` | Gets the amount of padding applied during the deconvolution operation. |
| `ParameterCount` | Gets a value indicating whether this layer supports training through backpropagation. |
| `Stride` | Gets the step size for positioning the kernel across the output data. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeActivationGradientGpuFallback(DirectGpuTensorEngine,Tensor<>,Tensor<>)` | Fallback activation gradient computation for unsupported GPU activation types. |
| `Forward(Tensor<>)` | Processes the input data through the deconvolutional layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass using fused ConvTranspose2D + Bias + Activation. |
| `GetMetadata` | Returns layer-specific metadata for serialization. |
| `GetParameterGradients` | Sets all trainable parameters of the layer from a single vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the kernel weights and biases with appropriate random values. |
| `OnFirstForward(Tensor<>)` | Resolves input shape on first forward (PyTorch ConvTranspose2d-style). |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters (kernel weights and biases) using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The bias values added to the deconvolution results for each output channel. |
| `_biasesGradient` | Calculated gradients for the biases during the backward pass. |
| `_kernels` | The collection of filter kernels used for the deconvolution operation. |
| `_kernelsGradient` | Calculated gradients for the kernels during the backward pass. |
| `_lastInput` | Stored input data from the most recent forward pass, used for backpropagation. |
| `_lastOutput` | Stored output data from the most recent forward pass, used for backpropagation. |

