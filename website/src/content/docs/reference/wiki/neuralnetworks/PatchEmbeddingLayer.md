---
title: "PatchEmbeddingLayer<T>"
description: "Implements a patch embedding layer for Vision Transformer (ViT) architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a patch embedding layer for Vision Transformer (ViT) architecture.

## For Beginners

This layer breaks an image into small square pieces (patches) and converts each patch
into a numerical representation that can be processed by a transformer.

Think of it like cutting a photo into a grid of smaller squares, then describing each square with numbers.
For example, a 224x224 image with 16x16 patches would be cut into 196 patches (14x14 grid), and each
patch would be represented by a vector of numbers (the embedding).

This allows transformers, which were originally designed for text, to process images by treating
the patches like "words" in a sentence.

## How It Works

The patch embedding layer divides an input image into fixed-size patches and projects them into an embedding space.
This is a key component of Vision Transformers, converting 2D spatial information into a sequence of embeddings
that can be processed by transformer encoder blocks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PatchEmbeddingLayer(Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new patch embedding layer with the specified dimensions. |
| `PatchEmbeddingLayer(Int32,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Eager-channel ctor: when the caller knows the input channel count at construction time (e.g. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of parameters in this layer. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Indicates whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildPatchEmbeddingOptimizerState(String)` | Builds the optimizer state for the specified parameter. |
| `EnsurePatchEmbeddingOptimizerState(IDirectGpuBackend,GpuOptimizerType)` | Ensures optimizer state buffers are allocated for the given optimizer type. |
| `Forward(Tensor<>)` | Performs the forward pass of the patch embedding layer. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass for patch embedding. |
| `GetParameterGradients` | Resets the internal state of the patch embedding layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all parameters of the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the weights and biases of the layer using Xavier initialization. |
| `OnFirstForward(Tensor<>)` | Resolves image height/width/channels from input on first forward (PyTorch-style). |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all parameters of the layer from a single vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the layer's parameters using the calculated gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates layer parameters using GPU-resident optimizer. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_channels` | The number of color channels in the input image (e.g., 3 for RGB). |
| `_embeddingDim` | The dimension of the embedding vector for each patch. |
| `_expectedInputChannels` | Expected input channels (3 for RGB, 1 for grayscale, etc.) when set via the eager-channel ctor. |
| `_imageHeight` | The height of the input image. |
| `_imageWidth` | The width of the input image. |
| `_lastInput` | Cached input from the forward pass for use in the backward pass. |
| `_lastPreActivation` | Cached pre-activation tensor from forward pass for use in activation derivative calculation. |
| `_numPatches` | The total number of patches (height x width). |
| `_numPatchesHeight` | The number of patches along the height dimension. |
| `_numPatchesWidth` | The number of patches along the width dimension. |
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |
| `_paramsLoadedViaSetParameters` | Gradients for projection weights calculated during backward pass. |
| `_patchSize` | The size of each square patch (both width and height). |
| `_projectionBias` | The bias terms added to the projected embeddings. |
| `_projectionBiasGradient` | Gradients for projection bias calculated during backward pass. |
| `_projectionWeights` | The projection weights that transform flattened patches to embeddings. |

