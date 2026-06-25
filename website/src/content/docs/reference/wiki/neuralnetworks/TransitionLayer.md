---
title: "TransitionLayer<T>"
description: "Implements a Transition Layer from the DenseNet architecture."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements a Transition Layer from the DenseNet architecture.

## For Beginners

The transition layer acts as a "bottleneck" between dense blocks.

Its purposes:

- Reduce feature map channels (compression): Dense blocks produce many channels
- Reduce spatial size (pooling): Helps control computational cost
- Improve model compactness without sacrificing accuracy

The compression factor (theta) controls how much to reduce channels.
theta=0.5 means halving the channels at each transition.

## How It Works

A Transition Layer is placed between Dense Blocks to reduce the number of feature maps
and spatial dimensions. It performs:

1. Batch Normalization
2. 1x1 Convolution (channel reduction by compression factor)
3. 2x2 Average Pooling with stride 2 (spatial dimension halving)

Architecture:

Where theta is the compression factor (default: 0.5).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransitionLayer(Double)` | Lazy ctor — input depth/height/width come from the first `Tensor{` call (`Tensor{`); only the channel-compression factor is required at construction. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` | Gets the number of output channels. |
| `ParameterCount` | Sum of trainable parameters across BN and the 1×1 projection. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AvgPool2DBackward(Tensor<>,Int32[])` | Backward pass for 4D average pooling. |
| `Forward(Tensor<>)` | Performs the forward pass of the Transition Layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass using GPU-resident tensors. |
| `GetParameters` | Gets all trainable parameters from the layer. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the layer. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from the given parameter vector. |
| `UpdateParameters()` | Updates the parameters of all sub-layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

