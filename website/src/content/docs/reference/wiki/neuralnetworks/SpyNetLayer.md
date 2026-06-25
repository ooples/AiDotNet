---
title: "SpyNetLayer<T>"
description: "SPyNet (Spatial Pyramid Network) layer for optical flow estimation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

SPyNet (Spatial Pyramid Network) layer for optical flow estimation.

## For Beginners

Optical flow tells us how pixels move between two frames.
SPyNet is a lightweight network that estimates this motion efficiently by
processing the images at multiple scales (pyramid levels).

The network works by:

1. Building image pyramids at different resolutions
2. Estimating flow at the coarsest level first
3. Refining the flow at each finer level
4. Combining all levels for the final flow

## How It Works

SPyNet uses a coarse-to-fine spatial pyramid approach to estimate optical flow
between two consecutive video frames. It's widely used in video super-resolution
and frame interpolation models.

**Reference:** Ranjan and Black, "Optical Flow Estimation using a Spatial Pyramid Network",
CVPR 2017. https://arxiv.org/abs/1611.00850

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpyNetLayer(Int32,IEngine)` | Creates a new SPyNet layer for optical flow estimation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsGpuExecution` | Indicates whether this layer supports GPU execution. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `Dispose(Boolean)` |  |
| `EstimateFlow(Tensor<>,Tensor<>)` | Estimates optical flow between two frames using separate tensors. |
| `Forward(Tensor<>)` |  |
| `ForwardGpu(Tensor<>[])` |  |
| `GetInputShape` |  |
| `GetOutputShape` | Gets the output shape for this layer (2 channels for optical flow: dx, dy). |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |
| `WarpImageWithGrid(Tensor<>,Tensor<>,Boolean)` | Warps an image using optical flow via IEngine.GridSample. |

