---
title: "ABME<T>"
description: "ABME: asymmetric bilateral motion estimation for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

ABME: asymmetric bilateral motion estimation for video frame interpolation.

## For Beginners

Most methods assume motion is symmetric (if something moves right
from frame 0, it moves left by the same amount from frame 1). But real motion isn't
symmetric -- a ball speeding up moves more in the second half. ABME estimates motion
independently in both directions, so it handles acceleration, deceleration, and curved
paths much better than symmetric methods.

**Usage:**

## How It Works

ABME (Park et al., ICCV 2021) uses asymmetric bilateral motion estimation:

- Bilateral motion estimation: estimates motion from the target time to both input frames

simultaneously (t to 0 and t to 1), rather than from input frames toward the target,
which is more natural for the interpolation task

- Asymmetric motion model: the two bilateral motion fields are NOT constrained to be

symmetric; each has its own magnitude and direction, correctly handling non-linear motion
paths (acceleration, deceleration, curved trajectories)

- Iterative GRU refinement: iteratively refines both bilateral flows with separate update

heads that can correct each flow independently over N iterations

- Context-aware synthesis: the final frame is synthesized by combining bilaterally warped

features with a learned blending mask that accounts for occlusion and motion boundaries

**Reference:** "Asymmetric Bilateral Motion Estimation for Video Frame Interpolation"
(Park et al., ICCV 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ABME(NeuralNetworkArchitecture<>,ABMEOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an ABME model in native training mode. |
| `ABME(NeuralNetworkArchitecture<>,String,ABMEOptions)` | Creates an ABME model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

