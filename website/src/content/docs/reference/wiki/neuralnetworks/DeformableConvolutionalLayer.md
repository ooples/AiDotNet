---
title: "DeformableConvolutionalLayer<T>"
description: "Deformable Convolutional Layer that learns spatial sampling offsets."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Deformable Convolutional Layer that learns spatial sampling offsets.

## For Beginners

Regular convolutions look at fixed grid positions around each pixel.
Deformable convolutions can look at shifted positions learned from the data itself.

This is useful for:

1. Aligning features between video frames (different objects may have moved differently)
2. Handling geometric transformations (rotation, scale, perspective)
3. Object detection with varying shapes and poses

## How It Works

Deformable convolution augments standard convolution with learnable 2D offsets for each
sampling location. This allows the convolution to adapt its receptive field to the
geometric structure of the input, making it particularly effective for video alignment.

**Reference:** Dai et al., "Deformable Convolutional Networks", ICCV 2017.
https://arxiv.org/abs/1703.06211

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeformableConvolutionalLayer(Int32,Int32,Int32,Int32,Int32,Int32,Boolean,IEngine)` | Creates a new Deformable Convolutional Layer with lazy spatial dims. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `EnsureDeformableOptimizerState(IDirectGpuBackend,GpuOptimizerType)` | Ensures GPU optimizer state buffers exist for all deformable convolution parameters. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors, keeping all data on GPU. |
| `GetInputShape` |  |
| `GetOutputShape` | Gets the output shape for this layer. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | GPU-resident parameter update with polymorphic optimizer support. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

