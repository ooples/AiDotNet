---
title: "ConvolutionalLayer<T>"
description: "Represents a convolutional layer in a neural network that applies filters to input data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a convolutional layer in a neural network that applies filters to input data.

## For Beginners

A convolutional layer is like a spotlight that scans over data
looking for specific patterns.

Think of it like examining a photo with a small magnifying glass:

- You move the magnifying glass across the image, one step at a time
- At each position, you note what you see in that small area
- After scanning the whole image, you have a collection of observations

For example, in image recognition:

- One filter might detect vertical edges
- Another might detect horizontal edges
- Together, they help the network recognize complex shapes

Convolutional layers are fundamental for recognizing patterns in images, audio, and other
grid-structured data.

## How It Works

A convolutional layer applies a set of learnable filters to input data to extract features. 
Each filter slides across the input data, performing element-wise multiplication and summing
the results. This operation is called convolution and is particularly effective for processing
grid-like data such as images.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConvolutionalLayer(Int32,Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>,IActivationFunction<>,Int32)` | Initializes a new instance of the `ConvolutionalLayer` class with the specified parameters and a scalar activation function. |
| `ConvolutionalLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `ConvolutionalLayer` class with the specified parameters and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Groups` | Number of convolution groups. |
| `InputDepth` | Gets the depth (number of channels) of the input data. |
| `IsDepthwise` | True when this layer is configured as a depthwise convolution. |
| `IsInitialized` |  |
| `KernelInChannels` | Per-group input-channel count = kernel's second dimension. |
| `KernelSize` | Gets the size of each filter (kernel) used in the convolution operation. |
| `OutputDepth` | Gets the depth (number of filters) of the output data. |
| `Padding` | Gets the amount of zero-padding added to the input data before convolution. |
| `ParameterCount` | Gets all trainable parameters of the layer as a single vector. |
| `Stride` | Gets the step size for moving the kernel across the input data. |
| `SupportsGpuExecution` | Gets whether this layer has a GPU implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyScalarActivationAutodiff(ComputationNode<>)` | Applies scalar activation function using autodiff operations. |
| `CalculateOutputDimension(Int32,Int32,Int32,Int32)` | Calculates the output dimension after applying a convolution operation. |
| `ComputeConvActivationGradientGpu(DirectGpuTensorEngine,Tensor<>,FusedActivationType)` | Computes activation gradient for convolutional layer using GPU-resident backward operations. |
| `Configure(Int32[],Int32,Int32,Int32,Int32,IActivationFunction<>)` | Creates a convolutional layer with the specified configuration using a fluent interface. |
| `Configure(Int32[],Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Creates a convolutional layer with the specified configuration and a vector activation function using a fluent interface. |
| `Deserialize(BinaryReader)` | Loads the layer's configuration and parameters from a binary reader. |
| `Dispose(Boolean)` | Releases resources used by this layer, including GPU tensor handles. |
| `Forward(Tensor<>)` | Processes the input data through the convolutional layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass using fused Conv2D + Bias + Activation. |
| `GetBiases` | Gets the biases tensor of the convolutional layer. |
| `GetFilters` | Gets a value indicating whether this layer supports training through backpropagation. |
| `GetMetadata` | Returns layer-specific metadata for serialization purposes. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` | Initializes the kernel weights and biases with random values. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` | Saves the layer's configuration and parameters to a binary writer. |
| `SetParameters(Vector<>)` | Sets all trainable parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters (kernel weights and biases) using the specified learning rate. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_addedBatchDimension` | Tracks whether a batch dimension was added during the forward pass. |
| `_biasReshaped4D` | Reference-keyed cache of the rank-1 `_biases` reshaped to `[1, OutputDepth, 1, 1]` for the conv-bias broadcast pattern. |
| `_biasReshaped4DSource` | Snapshot of the `_biases` reference at the moment `_biasReshaped4D` was populated. |
| `_biases` | The bias values added to the convolution results for each output channel. |
| `_biasesGradient` | Gradient of the biases computed during backpropagation via autodiff. |
| `_isInitialized` | Tracks whether lazy initialization has been completed. |
| `_kernels` | The collection of filter kernels used for the convolution operation. |
| `_kernelsGradient` | The execution engine for GPU-accelerated convolution operations. |
| `_lastInput` | Stored input data from the most recent forward pass, used for backpropagation. |
| `_lastOutput` | Stored output data from the most recent forward pass, used for backpropagation. |
| `_nonlinearityForInit` | Optional override for Kaiming init's gain. |
| `_originalInputShape` | Stores the original input shape for restoring higher-rank tensor output. |
| `_preAllocatedOutput` | Pre-allocated output buffer for Conv2DInto. |
| `_random` | Random number generator used for weight initialization. |

