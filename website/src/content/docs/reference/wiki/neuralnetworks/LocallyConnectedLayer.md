---
title: "LocallyConnectedLayer<T>"
description: "Represents a Locally Connected layer which applies different filters to different regions of the input, unlike a convolutional layer which shares filters."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Locally Connected layer which applies different filters to different regions of the input, unlike a convolutional layer which shares filters.

## For Beginners

This layer is like a specialized convolutional layer where each region gets its own unique filter.

Think of a Locally Connected layer like having specialized detectors for different regions:

- In a regular convolutional layer, the same filter slides across the entire input
- In a locally connected layer, each position has its own unique filter
- This means the layer can learn location-specific features

For example, in face recognition:

- A convolutional layer would use the same detector for eyes, whether looking at the top-left or bottom-right
- A locally connected layer would use different detectors depending on where it's looking

This specialization increases the model's power but:

- Requires more parameters
- May not generalize as well to new examples
- Is more computationally intensive

## How It Works

The Locally Connected layer is similar to a convolutional layer in that it applies filters to local regions 
of the input, but differs in that it uses different filter weights for each spatial location. This increases
the number of parameters and the expressiveness of the model, but reduces generalization capabilities.
It's useful when the patterns in different regions of the input are inherently different, such as in
face recognition where different parts of a face have different characteristics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LocallyConnectedLayer(Int32,Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `LocallyConnectedLayer` class with the specified dimensions, kernel parameters, and element-wise activation function. |
| `LocallyConnectedLayer(Int32,Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `LocallyConnectedLayer` class with the specified dimensions, kernel parameters, and vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildLocallyConnectedOptimizerState(String)` | Builds the optimizer state for a specific locally connected parameter. |
| `ComputeActivationGradientGpu(DirectGpuTensorEngine,Tensor<>,Tensor<>,FusedActivationType)` | Computes the activation gradient on GPU for locally connected layer backward pass. |
| `EnsureLocallyConnectedOptimizerState(IDirectGpuBackend,GpuOptimizerType)` | Ensures GPU optimizer state buffers exist for all locally connected parameters. |
| `Forward(Tensor<>)` | Performs the forward pass of the locally connected layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors, keeping all data on GPU. |
| `GetParameterGradients` | Sets the trainable parameters of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the layer. |
| `OnFirstForward(Tensor<>)` | Resolves spatial dims and allocates per-position weights on first forward. |
| `ResetState` | Resets the internal state of the layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters using GPU-based optimizer. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biasGradients` | Stores the gradients for the biases calculated during the backward pass. |
| `_biases` | The bias values for each output channel. |
| `_inputChannels` | The number of channels in the input tensor. |
| `_inputHeight` | The height of the input tensor. |
| `_inputWidth` | The width of the input tensor. |
| `_kernelSize` | The size of the kernel (filter) in both height and width dimensions. |
| `_lastInput` | Stores the input tensor from the last forward pass for use in the backward pass. |
| `_lastOutput` | Stores the output tensor from the last forward pass for use in the backward pass. |
| `_lastPreActivation` | Stores the pre-activation output from the last forward pass for use in the backward pass. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_outputChannels` | The number of channels in the output tensor. |
| `_outputHeight` | The height of the output tensor. |
| `_outputWidth` | The width of the output tensor. |
| `_stride` | The stride (step size) of the kernel when moving across the input. |
| `_weightGradients` | Stores the gradients for the weights calculated during the backward pass. |
| `_weights` | The weight tensors for the locally connected filters. |

