---
title: "OpticalFlowBase<T>"
description: "Base class for optical flow estimation models that compute dense pixel-wise motion between frames."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for optical flow estimation models that compute dense pixel-wise motion between frames.

## For Beginners

Optical flow tells you how each pixel moved between two frames.
It's like tracking every single point in the image. The output is a "flow field" where
each position stores (dx, dy) - how far that pixel moved horizontally and vertically.
This is useful for video stabilization, frame interpolation, action recognition, and more.

## How It Works

Optical flow estimation computes per-pixel motion vectors between two consecutive frames.
This base class provides:

- Flow field output (dense 2D displacement vectors)
- Multi-scale iterative refinement support
- Forward-backward consistency checking
- Flow visualization utilities

Derived classes implement specific architectures like RAFT, FlowFormer, SEA-RAFT, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpticalFlowBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the OpticalFlowBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumIterations` | Gets the number of iterative refinement steps. |
| `SupportsMultiScale` | Gets whether this model supports multi-scale processing. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEndpointError(Tensor<>,Tensor<>)` | Computes the endpoint error (EPE) between estimated and ground truth flow. |
| `ComputeForwardBackwardConsistency(Tensor<>,Tensor<>)` | Computes forward-backward consistency between two flow fields. |
| `EstimateFlow(Tensor<>,Tensor<>)` | Estimates optical flow between two frames. |
| `EstimateFlowMultiScale(Tensor<>,Tensor<>,Int32)` | Estimates optical flow at multiple scales for handling large motions. |
| `PredictCore(Tensor<>)` |  |

