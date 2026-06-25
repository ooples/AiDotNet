---
title: "FrameInterpolationBase<T>"
description: "Base class for frame interpolation models that generate intermediate frames between existing frames."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Video`

Base class for frame interpolation models that generate intermediate frames between existing frames.

## For Beginners

Frame interpolation makes choppy video smoother by adding new frames
in between existing ones. For example, converting 30fps video to 60fps or even 120fps.
The model "imagines" what the scene looks like at the intermediate time points.

## How It Works

Frame interpolation increases video frame rate by generating new frames between existing ones.
This base class provides:

- Arbitrary timestep interpolation (not just midpoint)
- Multi-frame interpolation (generate multiple intermediate frames)
- Flow-based and kernel-based interpolation utilities
- Temporal consistency support

Derived classes implement specific architectures like RIFE, AMT, EMA-VFI, VFIMamba, etc.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FrameInterpolationBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the FrameInterpolationBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsArbitraryTimestep` | Gets whether this model supports arbitrary timestep interpolation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` | Interpolates a single intermediate frame between two input frames at the given timestep. |
| `InterpolateMulti(Tensor<>,Tensor<>,Int32)` | Interpolates multiple intermediate frames between two input frames. |
| `InterpolateSequence(Tensor<>)` | Interpolates all frames in a video sequence to increase frame rate. |
| `PredictCore(Tensor<>)` |  |
| `TryGetArchitectureInputShape` | Frame-interpolation models consume two RGB frames concatenated channel-wise — 2 × Architecture.InputDepth = 6 channels — but Architecture.InputDepth itself reports the SINGLE-FRAME count (3) so it matches the architecture's per-frame metada… |

## Fields

| Field | Summary |
|:-----|:--------|
| `_temporalScaleFactor` | Gets the temporal scale factor (e.g., 2 for doubling frame rate). |

