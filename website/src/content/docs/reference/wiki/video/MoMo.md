---
title: "MoMo<T>"
description: "MoMo: momentum diffusion model for bi-directional optical flow in frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

MoMo: momentum diffusion model for bi-directional optical flow in frame interpolation.

## For Beginners

Most frame interpolation methods estimate motion (optical flow) using
direct prediction, which can be blurry at object edges. MoMo instead uses a generative AI
model to create sharper, more accurate motion fields, then uses those flows to produce
the intermediate frame. Think of it as using AI to "draw" better motion maps.

**Usage:**

## How It Works

MoMo (2024) is the first diffusion model for bi-directional optical flow in VFI:

- Flow diffusion: instead of directly regressing optical flow from a CNN (which produces

over-smoothed flow at boundaries), MoMo uses a denoising diffusion model to generate
bi-directional flow fields, capturing sharper motion boundaries and multi-modal flow

- Momentum-based flow modeling: incorporates a momentum prior that biases flow generation

toward physically plausible motions, reducing artifacts from unrealistic flow predictions

- Joint bi-directional generation: generates forward (t0 to t) and backward (t1 to t)

flows simultaneously in a single diffusion process, ensuring temporal consistency between
the two flow fields

- Flow-to-frame synthesis: the generated flows are used for backward warping with learned

occlusion masks and residual refinement to produce the final interpolated frame

**Reference:** "MoMo: Momentum Diffusion Model for Bi-Directional Flow" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoMo(NeuralNetworkArchitecture<>,MoMoOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MoMo model in native training mode. |
| `MoMo(NeuralNetworkArchitecture<>,String,MoMoOptions)` | Creates a MoMo model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

