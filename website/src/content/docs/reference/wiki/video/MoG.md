---
title: "MoG<T>"
description: "MoG: motion-aware generative frame interpolation combining flow and diffusion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

MoG: motion-aware generative frame interpolation combining flow and diffusion.

## For Beginners

MoG combines two approaches: first it figures out how things move
(optical flow), then uses a generative AI model (diffusion) to "paint" the intermediate
frame guided by that motion information. This produces sharper results than just blending
warped frames, especially for complex motions.

**Usage:**

## How It Works

MoG (2025) combines flow estimation with diffusion-based generation:

- Motion-aware conditioning: first estimates bidirectional optical flow using an EMA-VFI-style

flow network, then uses the estimated flows as spatial conditioning for a diffusion model
rather than directly warping frames

- Flow-conditioned diffusion: the denoising U-Net receives concatenated flow maps as

additional input channels, guiding the diffusion process to generate motion-consistent
intermediate frames with fine texture details

- Generative refinement: instead of blending warped frames (which can produce ghosting),

the diffusion model generates the intermediate frame from scratch, conditioned on the
input frames and estimated motion, producing sharp results even in occluded regions

- Progressive denoising: multi-step denoising with motion-aware noise scheduling that

preserves motion coherence in early steps and refines textures in later steps

**Reference:** "MoG: Motion-Aware Generative Frame Interpolation" (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoG(NeuralNetworkArchitecture<>,MoGOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a MoG model in native training mode. |
| `MoG(NeuralNetworkArchitecture<>,String,MoGOptions)` | Creates a MoG model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

