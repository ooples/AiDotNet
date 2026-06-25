---
title: "HighwayLayer<T>"
description: "Represents a Highway Neural Network layer that allows information to flow unchanged through the network."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Highway Neural Network layer that allows information to flow unchanged through the network.

## For Beginners

This layer helps solve a common problem in deep neural networks: difficulty in training very deep networks.

Think of the Highway Layer like a road with two lanes:

- The "transform lane" processes the data like a regular neural network layer
- The "bypass lane" lets information pass through unchanged
- A "gate" controls how much information flows through each lane

For example, when processing an image, the gate might let basic features like edges pass through
directly while sending more complex features through the transform lane for further processing.

This helps the network train more effectively because important information can flow more easily
through many layers without being lost or distorted.

## How It Works

A Highway Layer enables networks to effectively train even when they are very deep by introducing
"gating units" which learn to selectively pass or transform information. Unlike regular feed-forward
layers, highway layers have two "lanes": the transform lane that processes input data and the bypass
lane that passes information unchanged. The balance between these two lanes is controlled by a learned
gating mechanism.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HighwayLayer(Int32,IActivationFunction<>,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `HighwayLayer` class with the specified dimensions and element-wise activation functions. |
| `HighwayLayer(Int32,IVectorActivationFunction<>,IVectorActivationFunction<>)` | Initializes a new instance of the `HighwayLayer` class with the specified dimensions and vector activation functions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the auxiliary loss contribution. |
| `ParameterCount` | Gets the total number of trainable parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAuxiliaryLoss` | Gets or sets a value indicating whether auxiliary loss is enabled for this layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivation(Tensor<>,IActivationFunction<>,IVectorActivationFunction<>)` | Applies the appropriate activation function to the input tensor. |
| `ApplyActivationDerivative(Tensor<>,Tensor<>,IActivationFunction<>,IVectorActivationFunction<>)` | Applies the derivative of the appropriate activation function to the input tensor. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer based on gate balance regularization. |
| `Forward(Tensor<>)` | Performs the forward pass of the highway layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using FusedLinearGpu for efficient computation. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the layer. |
| `InitializeTensor(Tensor<>,)` | Initializes a 2D tensor with scaled random values. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gateActivation` | The element-wise activation function applied to the gate output. |
| `_gateBias` | The bias tensor added to the gate computation. |
| `_gateBiasGradient` | Stores the gradients for the gate bias calculated during the backward pass. |
| `_gateWeights` | The weight tensor used to compute the gate values. |
| `_gateWeightsGradient` | Stores the gradients for the gate weights calculated during the backward pass. |
| `_lastGateBalanceLoss` | Stores the last computed gate balance loss for diagnostic purposes. |
| `_lastGateOutput` | Stores the gate output tensor from the last forward pass for use in the backward pass. |
| `_lastGatePreActivation` | Stores the pre-activation gate values from the last forward pass. |
| `_lastInput` | Stores the input tensor from the last forward pass for use in the backward pass. |
| `_lastOutput` | Stores the output tensor from the last forward pass for use in the backward pass. |
| `_lastTransformOutput` | Stores the transformed output tensor from the last forward pass for use in the backward pass. |
| `_lastTransformPreActivation` | Stores the pre-activation transform values from the last forward pass. |
| `_transformActivation` | The element-wise activation function applied to the transform output. |
| `_transformBias` | The bias tensor added to the transformed input. |
| `_transformBiasGradient` | Stores the gradients for the transform bias calculated during the backward pass. |
| `_transformWeights` | The weight tensor used to transform the input data. |
| `_transformWeightsGradient` | Stores the gradients for the transform weights calculated during the backward pass. |
| `_vectorGateActivation` | The vector activation function applied to the gate output. |
| `_vectorTransformActivation` | The vector activation function applied to the transform output. |

