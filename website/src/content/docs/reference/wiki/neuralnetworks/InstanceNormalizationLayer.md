---
title: "InstanceNormalizationLayer<T>"
description: "Represents an Instance Normalization layer that normalizes each channel independently across spatial dimensions."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents an Instance Normalization layer that normalizes each channel independently across spatial dimensions.

## For Beginners

This layer helps stabilize training, especially for style transfer and image generation.

Think of Instance Normalization like adjusting the contrast of each color channel independently:

- Each channel (e.g., red, green, blue) is normalized on its own
- Each image in the batch is treated independently
- This removes instance-specific contrast information

Key advantages:

- Works well for style transfer and image generation
- Independent of batch size (works with batch size of 1)
- Removes instance-specific style information, making it ideal for style transfer

Common usage:

- Style transfer networks (separates content from style)
- GANs (Generative Adversarial Networks)
- Image-to-image translation

## How It Works

Instance Normalization normalizes each channel independently for each sample in the batch.
Unlike Batch Normalization which computes statistics across the batch dimension,
Instance Normalization computes statistics independently for each sample and each channel.
This is essentially Group Normalization with numGroups = numChannels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstanceNormalizationLayer(Int32,Double,Boolean)` | Initializes a new instance of the InstanceNormalizationLayer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Affine` | Gets whether affine transformation (learnable gamma and beta) is enabled. |
| `NumChannels` | Gets the number of channels this layer normalizes. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of instance normalization. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass of instance normalization on GPU tensors. |
| `GetBeta` | Gets the beta (shift) parameters. |
| `GetBetaTensor` | Gets the beta (shift) parameters as a tensor. |
| `GetEpsilon` | Gets the epsilon value used for numerical stability. |
| `GetGamma` | Gets the gamma (scale) parameters. |
| `GetGammaTensor` | Gets the gamma (scale) parameters as a tensor. |
| `GetMetadata` | Gets metadata for serialization. |
| `GetParameterGradients` | Resets the internal state of the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

