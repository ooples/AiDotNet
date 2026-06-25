---
title: "ConvLSTMLayer<T>"
description: "Implements a Convolutional Long Short-Term Memory (ConvLSTM) layer for processing sequential spatial data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a Convolutional Long Short-Term Memory (ConvLSTM) layer for processing sequential spatial data.

## For Beginners

ConvLSTM is like a smart video analyzer that remembers spatial patterns over time.

Imagine you're watching a video of clouds moving across the sky:

1. ConvLSTM looks at each frame (like a photo) in the video sequence
2. It remembers important spatial features (like cloud shapes) from previous frames
3. It uses this memory to predict how these features might change in future frames

This layer is particularly good at:

- Predicting what might happen next in a video
- Analyzing patterns in weather maps over time
- Understanding how spatial arrangements change in a sequence

Unlike simpler layers that treat each frame independently, ConvLSTM connects the dots
between frames, making it powerful for tasks involving moving images or changing spatial data.

## How It Works

ConvLSTM combines convolutional operations with LSTM (Long Short-Term Memory) to handle
spatial-temporal data. It's particularly useful for tasks involving sequences of images or
spatial data, such as video prediction, weather forecasting, and spatiotemporal sequence prediction.

Key features of ConvLSTM:

- Maintains spatial information throughout the processing
- Captures both spatial and temporal dependencies
- Uses convolutional operations instead of matrix multiplications in the LSTM cell
- Suitable for data with both spatial and temporal structure

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConvLSTMLayer(Int32[],Int32,Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the ConvLSTMLayer class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | The computation engine (CPU or GPU) for vectorized operations. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU-accelerated forward pass. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivationDerivative(Tensor<>)` | Applies the derivative of the activation function to the input tensor. |
| `BackwardStep(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Performs the backward step for a single time step in the ConvLSTM layer. |
| `BuildConvLstmOptimizerState(String)` | Builds the optimizer state for a specific ConvLSTM parameter. |
| `CalculateOutputShape(Int32[],Int32,Int32,Int32,Int32)` | Calculates the output shape of the layer based on input dimensions and layer parameters. |
| `ClearGpuCache` | Clears all cached GPU buffers from the forward pass. |
| `ConvLSTMCell(Tensor<>,Tensor<>,Tensor<>)` | Processes a single time step through the ConvLSTM cell. |
| `Convolve(Tensor<>,Tensor<>)` | Performs a 2D convolution operation between an input tensor and a kernel. |
| `CopyTensorToVector(Tensor<>,Vector<>,Int32)` | Helper method to copy values from a tensor to a vector. |
| `CopyVectorToTensor(Vector<>,Tensor<>,Int32)` | Helper method to copy values from a vector to a tensor. |
| `EnsureConvLstmOptimizerState(IDirectGpuBackend,GpuOptimizerType)` | Ensures GPU optimizer state buffers exist for all ConvLSTM parameters. |
| `Forward(Tensor<>)` | Performs the forward pass of the ConvLSTM layer. |
| `ForwardGpu(Tensor<>[])` | Performs a GPU-resident forward pass of the ConvLSTM layer. |
| `ForwardStep(Tensor<>,Tensor<>,Tensor<>)` | Performs a single forward step of the ConvLSTM cell for one time step. |
| `GetMetadata` | Persists kernelSize / filters / padding / strides so the deser path doesn't need to fabricate them. |
| `GetParameterGradients` | Sets all trainable parameters of the ConvLSTM layer from a flattened vector. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Retrieves all trainable parameters of the ConvLSTM layer as a flattened vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeBiasesToZero(Tensor<>)` | Initializes all bias values to zero. |
| `InitializeWeights(Tensor<>)` | Initializes the weights of the layer with small random values. |
| `ResetState` | Resets the internal state of the ConvLSTM layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameterWithMomentum(Tensor<>,String,)` | Updates a single parameter tensor using gradient descent with momentum. |
| `UpdateParameters()` | Updates all trainable parameters of the layer using the computed gradients and specified learning rate. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | GPU-resident parameter update with polymorphic optimizer support. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gpuBatchSize` | GPU state dimensions cached during forward pass. |
| `_gpuCandidateCells` | Cached candidate cell values for each timestep. |
| `_gpuCellStates` | Cached cell states for each timestep - needed for cell gradient computation. |
| `_gpuForgetGates` | Cached forget gate values for each timestep. |
| `_gpuHiddenStates` | Cached hidden states for each timestep - needed for hidden weight gradients. |
| `_gpuInput` | Cached GPU input for backward pass. |
| `_gpuInputGates` | Cached input gate values for each timestep. |
| `_gpuInputShape` | Cached GPU input shape for backward pass. |
| `_gpuInputSlices` | Cached input slices (NCHW) for each timestep - needed for weight gradients. |
| `_gpuOutputGates` | Cached output gate values for each timestep. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

