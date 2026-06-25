---
title: "InvertedResidualBlock<T>"
description: "Implements an Inverted Residual Block (MBConv) used in MobileNetV2 and MobileNetV3."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements an Inverted Residual Block (MBConv) used in MobileNetV2 and MobileNetV3.

## For Beginners

The Inverted Residual Block is designed for efficient mobile inference.

Key innovations:

- Expansion: First EXPANDS the channels (opposite of traditional bottlenecks)
- Depthwise separable convolution: Filters each channel independently, then mixes
- Linear bottleneck: The final projection has NO activation (preserves information)
- Skip connection: Only when input and output dimensions match

This design reduces computational cost while maintaining model accuracy.

## How It Works

The Inverted Residual Block is the core building block for MobileNet architectures.
Unlike traditional residual blocks that narrow then widen, inverted residual blocks
expand then narrow, hence the name "inverted."

Architecture:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvertedResidualBlock(Int32,Int32,Int32,Boolean,Int32,IActivationFunction<>)` | Initializes a new instance of the `InvertedResidualBlock` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpansionRatio` | Gets the expansion ratio. |
| `InChannels` | Gets the number of input channels. |
| `OutChannels` | Gets the number of output channels. |
| `ParameterCount` | Sum of trainable parameters across all sub-layers. |
| `Stride` | Gets the stride used in the depthwise convolution. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer has a GPU implementation. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise using vectorized Engine operations. |
| `Forward(Tensor<>)` | Performs the forward pass of the Inverted Residual Block. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU, keeping data GPU-resident. |
| `GetMetadata` | Returns layer-specific metadata for serialization purposes. |
| `GetParameters` | Gets all trainable parameters from the block. |
| `MultiplyTensors(Tensor<>,Tensor<>)` | Multiplies two tensors element-wise using vectorized Engine operations. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the block. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from the given parameter vector. |
| `SetTrainingMode(Boolean)` |  |
| `TransposeNCHWToNHWC(Tensor<>)` | Transposes a tensor from NCHW to NHWC format. |
| `TransposeNHWCToNCHW(Tensor<>)` | Transposes a tensor from NHWC to NCHW format. |
| `UpdateParameters()` | Updates the parameters of all sub-layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_pendingExtraParameters` | Buffer for ILayerSerializationExtras.SetExtraParameters when called pre-OnFirstForward. |

