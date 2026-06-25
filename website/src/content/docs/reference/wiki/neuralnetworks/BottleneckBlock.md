---
title: "BottleneckBlock<T>"
description: "Implements the BottleneckBlock used in ResNet50, ResNet101, and ResNet152 architectures."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements the BottleneckBlock used in ResNet50, ResNet101, and ResNet152 architectures.

## For Beginners

The BottleneckBlock is like a compressed processing pipeline.

Think of it as:

1. First 1x1 conv: "Compress" - reduce the number of channels (like compressing a file)
2. 3x3 conv: "Process" - do the heavy computation on the compressed representation
3. Second 1x1 conv: "Expand" - restore and expand the channels

This is more efficient because:

- The expensive 3x3 convolution works on fewer channels
- The overall result has high capacity (4x expansion)
- Much fewer parameters than three 3x3 convolutions

The expansion factor of 4 means if the base channels is 64, the output will have 256 channels.

## How It Works

The BottleneckBlock uses a 1x1-3x3-1x1 convolution pattern, where the 1x1 layers reduce and then
restore dimensions (with expansion), and the 3x3 layer is the bottleneck with smaller channels.
This design is more computationally efficient than stacking 3x3 convolutions for deep networks.

**Architecture:**
The first 1x1 conv reduces channels, the 3x3 processes at reduced channels,
and the final 1x1 expands channels by a factor of 4.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BottleneckBlock(Int32,Int32,Boolean)` | Initializes a new instance of the `BottleneckBlock` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer has a GPU implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through the BottleneckBlock. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU, keeping data GPU-resident. |
| `GetParameters` | Gets all trainable parameters. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the block. |
| `UpdateParameters()` | Updates the parameters of all internal layers. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Expansion` | The expansion factor for BottleneckBlock. |

