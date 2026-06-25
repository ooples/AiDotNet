---
title: "DynamiCrafter<T>"
description: "DynamiCrafter: animating open-domain images with video diffusion priors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

DynamiCrafter: animating open-domain images with video diffusion priors.

## For Beginners

DynamiCrafter uses an AI video generator (diffusion model) that
already understands how things move in the real world. Given a start frame and end frame,
it gradually "imagines" what happens in between, producing natural-looking intermediate
frames with realistic motion, lighting changes, and object interactions.

**Usage:**

## How It Works

DynamiCrafter (2024) uses video diffusion priors for frame interpolation:

- Video diffusion backbone: adapts a pre-trained text-to-video diffusion model for the

interpolation task, leveraging its learned motion priors from millions of training videos

- First/last frame conditioning: the diffusion process is conditioned on both endpoint

frames using CLIP image embeddings injected via cross-attention, ensuring temporal
consistency with both the start and end frames

- Noise schedule adaptation: modified diffusion noise schedule that biases early denoising

steps toward global motion consistency and later steps toward fine detail refinement

- Temporal attention: 3D self-attention across generated frames ensures smooth motion

transitions without flickering or temporal discontinuities

**Reference:** "DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynamiCrafter(NeuralNetworkArchitecture<>,DynamiCrafterOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DynamiCrafter model in native training mode. |
| `DynamiCrafter(NeuralNetworkArchitecture<>,String,DynamiCrafterOptions)` | Creates a DynamiCrafter model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

