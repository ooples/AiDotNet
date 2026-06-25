---
title: "MoGOptions"
description: "Configuration options for MoG motion-aware generative frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for MoG motion-aware generative frame interpolation.

## For Beginners

MoG combines two approaches: first it figures out how things move
(optical flow), then uses a generative AI model (diffusion) to "paint" the intermediate
frame guided by that motion information. This produces sharper results than just blending
warped frames, especially for complex motions.

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

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoGOptions` | Initializes a new instance with default values. |
| `MoGOptions(MoGOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `GuidanceScale` | Gets or sets the guidance scale for classifier-free guidance. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion denoising steps during inference. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumFlowScales` | Gets or sets the number of flow estimation scales. |
| `NumResBlocks` | Gets or sets the number of U-Net residual blocks per level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

