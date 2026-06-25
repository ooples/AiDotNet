---
title: "DilatedConvolutionalLayer<T>"
description: "Represents a dilated convolutional layer for neural networks that applies filters with gaps between filter elements."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a dilated convolutional layer for neural networks that applies filters with gaps between filter elements.

## For Beginners

A dilated convolutional layer is like looking at an image with a special magnifying glass.

Regular convolutions look at pixels that are right next to each other, like this:

- Looking at a 3×3 area of an image (9 adjacent pixels)

Dilated convolutions skip some pixels, creating gaps, like this:

- With dilation=2, it looks at pixels with a gap of 1 pixel between them
- The 3×3 filter now covers a 5×5 area (still using only 9 values)

Benefits:

- Sees a larger area without needing more computing power
- Captures wider patterns in the data
- Helps detect features at different scales

For example, in image processing, dilated convolutions can help the network understand 
both fine details and broader context at the same time.

## How It Works

A dilated convolutional layer extends traditional convolutional layers by introducing gaps (dilation) between 
the elements of the convolution kernel. This increases the receptive field without increasing the number of 
parameters or computational cost. Dilated convolutions are particularly useful in tasks requiring a wide 
context without sacrificing resolution, such as semantic segmentation or audio generation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DilatedConvolutionalLayer(Int32,Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `DilatedConvolutionalLayer` class with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateInputShape(Int32,Int32,Int32)` | Calculates the shape of the input tensor expected by this layer. |
| `CalculateOutputDimension(Int32,Int32,Int32,Int32,Int32)` | Calculates the output dimension (height or width) based on input dimension and convolution parameters. |
| `CalculateOutputShape(Int32,Int32,Int32)` | Calculates the shape of the output tensor produced by this layer. |
| `Forward(Tensor<>)` | Performs the forward pass of the convolutional layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass using fused DilatedConv2D + Bias + Activation. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeWeights` | Initializes the kernel weights and biases with appropriate values. |
| `OnFirstForward(Tensor<>)` | Resolves input depth and output spatial dims on first forward. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's weights and biases using the calculated gradients and the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Indicates whether a batch dimension was added during forward pass. |
| `_biasGradients` | The gradients for the biases, computed during backpropagation. |
| `_biases` | The bias values for each output channel. |
| `_dilation` | The dilation factor that determines the spacing between kernel elements. |
| `_inputDepth` | The number of channels in the input data. |
| `_kernelGradients` | The gradients for the kernels, computed during backpropagation. |
| `_kernelSize` | The size of the convolution kernel (filter) in both height and width dimensions. |
| `_kernels` | The weight tensors (filters) for the convolutional layer. |
| `_lastInput` | The input tensor from the last forward pass, saved for backpropagation. |
| `_lastOutput` | The output tensor from the last forward pass, saved for backpropagation. |
| `_originalInputShape` | Stores the original input shape before any reshaping. |
| `_outputDepth` | The number of filters (output channels) in the convolutional layer. |
| `_padding` | The amount of zero-padding added to the input before convolution. |
| `_stride` | The stride (step size) of the convolution operation. |

