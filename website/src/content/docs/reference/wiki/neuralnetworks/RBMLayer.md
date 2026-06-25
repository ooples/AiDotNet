---
title: "RBMLayer<T>"
description: "Represents a Restricted Boltzmann Machine (RBM) layer for neural networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Restricted Boltzmann Machine (RBM) layer for neural networks.

## For Beginners

An RBM layer is like a feature detector that can learn patterns in data.

Imagine you have a set of movie ratings:

- The visible layer represents the actual ratings
- The hidden layer represents abstract features (e.g., "likes action", "prefers comedy")
- The RBM learns to connect ratings to these abstract features

RBM layers are useful for:

- Finding underlying patterns in data
- Reducing the dimensionality of data
- Initializing weights for deep neural networks

## How It Works

An RBM layer is a stochastic neural network layer that learns a probability distribution over its inputs.
It consists of a visible layer and a hidden layer with no connections between nodes within the same layer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RBMLayer(Int32,IActivationFunction<>)` | Lazy ctor — visible-unit count is resolved from `input.Shape[^1]` on the first `Forward` call. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInitialized` |  |
| `ParameterCount` | Gets the total number of trainable parameters in the layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Indicates whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOuterProduct(Vector<>,Vector<>)` | Computes the outer product of two vectors as a 2D tensor. |
| `ComputeOuterProductTensor(Tensor<>,Tensor<>)` | Computes the outer product of two tensors as a 2D tensor. |
| `EnsureInitialized` |  |
| `Forward(Tensor<>)` | Computes the forward pass of the RBM layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using FusedLinearGpu with Sigmoid activation. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the RBM layer. |
| `OnFirstForward(Tensor<>)` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SampleBinaryStates(Vector<>)` | Samples binary states (0 or 1) from probability values. |
| `SampleBinaryStatesTensor(Tensor<>)` | Samples binary states from probability tensor using stochastic sampling. |
| `SampleHiddenGivenVisible(Vector<>)` | Computes the probability of each hidden unit being active given the visible units. |
| `SampleHiddenGivenVisibleTensor(Tensor<>)` | Computes the probability of each hidden unit being active given the visible units (tensor-based). |
| `SampleVisibleGivenHidden(Vector<>)` | Computes the probability of each visible unit being active given the hidden units. |
| `SampleVisibleGivenHiddenTensor(Tensor<>)` | Computes the probability of each visible unit being active given the hidden units (tensor-based). |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `TrainWithContrastiveDivergence(Vector<>,,Int32)` | Trains the RBM using contrastive divergence with the given data. |
| `TrainWithContrastiveDivergenceTensor(Tensor<>,,Int32)` | Tensor-based contrastive divergence training - no type conversions in hot path. |
| `UpdateParameters()` | Updates the layer's parameters using either standard backpropagation or contrastive divergence. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hiddenBiases` | Gets or sets the bias values for the hidden units. |
| `_hiddenBiasesGradient` | Gradient of the hidden biases computed during backpropagation. |
| `_hiddenUnits` | Gets the number of units in the hidden layer. |
| `_isInitialized` | Tracks whether weights / biases have been materialized. |
| `_lastHiddenOutput` | Stores the last output from the hidden layer during training. |
| `_lastVisibleInput` | Stores the last input from the visible layer during training. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_reconstructedHidden` | Stores the reconstructed hidden layer activations during training. |
| `_reconstructedVisible` | Stores the reconstructed visible layer activations during training. |
| `_visibleBiases` | Gets or sets the bias values for the visible units. |
| `_visibleBiasesGradient` | Gradient of the visible biases computed during backpropagation. |
| `_visibleUnits` | Gets the number of units in the visible layer. |
| `_weights` | Gets or sets the weight matrix connecting visible and hidden units. |
| `_weightsGradient` | Gradient of the weights computed during backpropagation. |

