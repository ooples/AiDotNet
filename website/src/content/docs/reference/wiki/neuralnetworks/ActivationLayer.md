---
title: "ActivationLayer<T>"
description: "A layer that applies an activation function to transform the input data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A layer that applies an activation function to transform the input data.

Activation functions introduce non-linearity to neural networks. Non-linearity means the output isn't 
simply proportional to the input (like y = 2x). Instead, it can follow curves or more complex patterns.
severely limiting what it can learn.

Common activation functions include:

- ReLU: Returns 0 for negative inputs, or the input value for positive inputs
- Sigmoid: Squashes values between 0 and 1, useful for probabilities
- Tanh: Similar to sigmoid but outputs values between -1 and 1

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsGpuExecution` | Gets whether this layer's activation function supports GPU execution. |
| `SupportsTraining` | Indicates whether this layer has trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyScalarActivation(Tensor<>)` | Applies a scalar activation function to each element of the input tensor. |
| `ApplyVectorActivation(Tensor<>)` | Applies a vector activation function to the entire input tensor. |
| `ConvertToOnnx(OnnxGraphBuilder,OnnxLayerInputs)` | Emits a single ONNX op corresponding to the layer's activation function. |
| `Forward(Tensor<>)` | Processes the input data by applying the activation function. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using GPU-accelerated activation kernels. |
| `GetActivationTypeName` | Gets the name of the activation function type for error messages. |
| `GetParameters` | Gets all trainable parameters of this layer as a flat vector. |
| `OnFirstForward(Tensor<>)` | Resolves shape on first forward; output equals input (passthrough). |
| `ResetState` | Clears the layer's memory of previous inputs. |
| `TryGetFusedActivationType(FusedActivationType)` | Attempts to map the configured activation function to a FusedActivationType for GPU execution. |
| `UpdateParameters()` | Updates the layer's internal parameters during training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInput` | Stores the input from the most recent forward pass for use in the backward pass. |
| `_useVectorActivation` | Indicates whether this layer uses a vector activation function instead of a scalar one. |

