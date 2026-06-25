---
title: "Conv3DLayer<T>"
description: "Represents a 3D convolutional layer for processing volumetric data like voxel grids."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a 3D convolutional layer for processing volumetric data like voxel grids.

## For Beginners

A 3D convolutional layer is like a 2D convolution but extended
to work with volumetric data.

Think of it like examining a 3D cube of data:

- A 2D convolution slides a filter across height and width
- A 3D convolution slides a filter across depth, height, and width

This is useful for:

- Recognizing 3D shapes from voxel grids (like ModelNet40)
- Analyzing medical scans (CT, MRI)
- Processing video frames as a 3D volume

The layer learns to detect 3D patterns like edges, surfaces, and volumes.

## How It Works

A 3D convolutional layer applies learnable filters to volumetric input data to extract
spatial features across all three dimensions. This is essential for processing 3D data
such as voxelized point clouds, medical imaging (CT/MRI), or video sequences.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Conv3DLayer(Int32,Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `Conv3DLayer` class with specified parameters. |
| `Conv3DLayer(Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `Conv3DLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input channels expected by this layer. |
| `KernelSize` | Gets the size of the 3D convolution kernel (same for depth, height, width). |
| `OutputChannels` | Gets the number of output channels (filters) produced by this layer. |
| `Padding` | Gets the zero-padding applied to all sides of the input volume. |
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `Stride` | Gets the stride of the convolution (step size when sliding the kernel). |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training (backpropagation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBiases(Tensor<>)` | Adds bias values to each output channel using vectorized operations. |
| `CalculateInputShape(Int32,Int32,Int32,Int32)` | Calculates the input shape array for the layer. |
| `CalculateOutputShape(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Calculates the output shape array based on convolution parameters. |
| `Clone` | Creates a deep copy of the layer with the same configuration and parameters. |
| `ComputeBiasGradient(Tensor<>)` | Computes the bias gradient by summing gradients over batch and spatial dimensions. |
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `Forward(Tensor<>)` | Performs the forward pass of the 3D convolution operation. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors, keeping all data on GPU. |
| `ForwardIm2Col(Tensor<>,FusedActivationType)` | Tape-recording forward path that replaces the engine's direct `Conv3D` with an im2col + GEMM formulation. |
| `GetBiases` | Gets the bias tensor. |
| `GetFilters` | Gets the convolution filter kernels. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the kernel weights tensor. |
| `InitializeWeights` | Initializes the kernel weights using He (Kaiming) initialization and biases to zero. |
| `OnFirstForward(Tensor<>)` | Resolves input shape (channels, depth, height, width) on first forward (PyTorch-style). |
| `ResetState` | Resets the cached state from forward/backward passes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer parameters using the computed gradients and learning rate. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters on GPU using the configured optimizer. |
| `ValidateParameters(Int32,Int32,Int32,Int32,Int32,Int32,Int32)` | Validates constructor parameters. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | The learnable bias values with shape [OutputChannels], one per output channel. |
| `_biasesGradient` | Cached gradient for biases computed during backward pass. |
| `_inputDepth` | Depth of input volume (cached for shape calculations). |
| `_inputHeight` | Height of input volume (cached for shape calculations). |
| `_inputWidth` | Width of input volume (cached for shape calculations). |
| `_kernels` | The learnable convolution kernels with shape [OutputChannels, InputChannels, KernelSize, KernelSize, KernelSize]. |
| `_kernelsGradient` | Cached gradient for kernels computed during backward pass. |
| `_lastInput` | Cached input from the last forward pass, needed for backward computation. |
| `_lastOutput` | Cached output from the last forward pass (after activation). |
| `_lastPreActivation` | Cached output from the last forward pass (before activation), needed for backward computation. |
| `_originalInputShape` | Original input shape for restoring higher-rank tensors after processing. |

