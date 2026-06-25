---
title: "SpectralNormalizationLayer<T>"
description: "Represents a spectral normalization layer that normalizes the weights of a layer by their spectral norm."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a spectral normalization layer that normalizes the weights of a layer by their spectral norm.

## For Beginners

Spectral normalization keeps layer weights from getting too large.

Key benefits:

- Stabilizes GAN training by preventing extreme weight values
- Ensures the discriminator doesn't become too powerful too quickly
- Helps prevent mode collapse in GANs
- Computationally efficient compared to other normalization methods

How it works:

- Computes the largest singular value of the weight matrix
- Divides all weights by this value
- Keeps weights normalized throughout training

Reference: Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)

## How It Works

Spectral normalization is a weight normalization technique that constrains the Lipschitz constant
of a neural network layer. It does this by dividing the weight matrix by its largest singular value
(spectral norm). This technique is particularly effective for stabilizing GAN training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralNormalizationLayer(ILayer<>,Int32)` | Initializes a new instance of the `SpectralNormalizationLayer` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU-resident training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSpectralNorm(Tensor<>)` | Computes the spectral norm using power iteration with vectorized operations. |
| `EnsureGpuPowerIterationVectors(DirectGpuTensorEngine,Int32,Int32)` | Initializes or reinitializes the GPU power iteration vectors when dimensions change. |
| `EnsurePowerIterationVectors(Int32,Int32)` | Initializes or reinitializes the power iteration vectors when dimensions change. |
| `Forward(Tensor<>)` | Performs the forward pass through the layer with spectrally normalized weights. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors with GPU-accelerated spectral normalization. |
| `GetBiasCount(Int32)` | Estimates the number of bias parameters, if present. |
| `GetMetadata` | Persists the inner layer's type name + shape and the power-iteration count so DeserializationHelper can reconstruct the wrapped layer concretely. |
| `GetParameterGradients` | Gets the parameter gradients from the inner layer. |
| `GetParameters` | Gets the parameters of the inner layer. |
| `NormalizeVector(Tensor<>)` | Normalizes a vector tensor in-place using Engine operations. |
| `ResetState` | Resets the internal state of the layer. |
| `RestoreOriginalWeights` | Restores the original weights after Backward or on exception. |
| `SetParameters(Vector<>)` | Sets the parameters of the inner layer. |
| `UpdateParameters()` | Updates the parameters of the inner layer. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | GPU-resident parameter update using the provided optimizer configuration. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_epsilon` | Epsilon value for numerical stability. |
| `_innerLayer` | The underlying layer whose weights will be normalized. |
| `_lastInput` | Cached input from the last forward pass. |
| `_lastOutput` | Cached output from the last forward pass. |
| `_normalizedWeightsApplied` | Flag indicating that normalized weights are currently applied. |
| `_originalParameters` | Original weights stored during Forward, to be restored after Backward. |
| `_powerIterations` | The number of power iterations to perform when computing the spectral norm. |
| `_u` | The left singular vector used for power iteration to compute the spectral norm. |
| `_uGpu` | GPU-resident power iteration vectors. |
| `_v` | The right singular vector used for power iteration. |

