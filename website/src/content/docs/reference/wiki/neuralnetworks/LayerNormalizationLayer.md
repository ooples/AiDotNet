---
title: "LayerNormalizationLayer<T>"
description: "Represents a Layer Normalization layer that normalizes inputs across the feature dimension."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Layer Normalization layer that normalizes inputs across the feature dimension.

## For Beginners

This layer helps stabilize and speed up training by standardizing the data.

Think of Layer Normalization like standardizing test scores:

- It makes each sample's features have a mean of 0 and standard deviation of 1
- It does this independently for each sample (unlike Batch Normalization)
- It applies this normalization along the feature dimension
- After normalizing, it scales and shifts the values using learnable parameters

For example, in a sentiment analysis task, some input sentences might use very positive words while 
others use more neutral language. Layer Normalization helps the network focus on the relative importance 
of features within each sample rather than their absolute values.

This is particularly useful for:

- Recurrent neural networks
- Cases where batch sizes are small
- Making training more stable and faster

## How It Works

Layer Normalization is a technique used to normalize the inputs to a layer, which can help improve
training stability and speed. Unlike Batch Normalization which normalizes across the batch dimension,
Layer Normalization normalizes across the feature dimension independently for each sample. This makes
it particularly useful for recurrent networks and when batch sizes are small. The layer learns scale
(gamma) and shift (beta) parameters to allow the network to recover the original representation if needed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerNormalizationLayer(Double)` | Initializes a new instance of the `LayerNormalizationLayer` class with the specified feature size and epsilon value. |
| `LayerNormalizationLayer(Int32,Double)` | AiDotNet#1370 eager-init constructor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets all trainable parameters of the layer as a single vector. |
| `SupportsGpuExecution` | Indicates whether this layer supports GPU-resident execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the layer normalization layer. |
| `ForwardGpu(Tensor<>[])` | GPU-resident forward pass for layer normalization. |
| `GetBeta` | Gets the beta (shift) parameters of the layer normalization layer. |
| `GetBetaTensor` | Gets the beta tensor for JIT compilation and internal use. |
| `GetEpsilon` | Gets the epsilon value used for numerical stability. |
| `GetGamma` | Gets the gamma (scale) parameters of the layer normalization layer. |
| `GetGammaTensor` | Gets the gamma tensor for JIT compilation and internal use. |
| `GetMetadata` | Returns layer-specific metadata required for cloning/serialization. |
| `GetNormalizedShape` | Gets the normalized shape (feature size) of the layer. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` | Resolves `featureSize` from `input.Shape[^1]` (last dim) on the first forward call, allocates gamma/beta tensors, and registers them as trainable parameters. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_beta` | The shift parameters learned during training. |
| `_betaGradient` | Stores the gradients for the beta parameters calculated during the backward pass. |
| `_epsilon` | A small value added to the variance for numerical stability. |
| `_gamma` | The scale parameters learned during training. |
| `_gammaGradient` | Stores the gradients for the gamma parameters calculated during the backward pass. |
| `_lastInput` | Stores the input tensor from the last forward pass for use in the backward pass. |
| `_lastMean` | Stores the mean values for each sample from the last forward pass. |
| `_lastVariance` | Stores the variance values for each sample from the last forward pass. |

