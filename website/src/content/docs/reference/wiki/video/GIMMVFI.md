---
title: "GIMMVFI<T>"
description: "GIMM-VFI: generalizable implicit motion modeling for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

GIMM-VFI: generalizable implicit motion modeling for video frame interpolation.

## For Beginners

GIMM-VFI learns a smooth "motion field" that describes how everything
in the scene moves over time. Once it processes two frames, it can generate a new frame at
ANY point between them. This is great for variable slow-motion effects or non-uniform
frame rate conversion where you need different time spacings between frames.

**Usage:**

## How It Works

GIMM-VFI (NeurIPS 2024) uses implicit neural representations for continuous-time motion:

- Implicit motion function: learns a continuous function M(x, y, t) that maps any spatial

position (x, y) and any timestep t in [0, 1] to a motion vector, enabling interpolation
at arbitrary time intervals without retraining or additional forward passes

- Motion encoding network: encodes two input frames into a shared motion latent space

using cross-correlation features, producing a motion representation that can be queried
at any desired timestep

- Generalizable across timesteps: a single forward pass through the motion encoder produces

a representation that the implicit function can query at any t, unlike methods that need
separate inference per timestep

- Adaptive sampling: the implicit function can be queried at higher density in regions with

complex motion and lower density in static regions for efficient computation

**Reference:** "GIMM-VFI: Generalizable Implicit Motion Modeling for Video Frame
Interpolation" (NeurIPS 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GIMMVFI(NeuralNetworkArchitecture<>,GIMMVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a GIMM-VFI model in native training mode. |
| `GIMMVFI(NeuralNetworkArchitecture<>,String,GIMMVFIOptions)` | Creates a GIMM-VFI model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

