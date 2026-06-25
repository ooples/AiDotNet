---
title: "SeedVR<T>"
description: "SeedVR: seeding infinity in diffusion transformer towards generic video restoration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

SeedVR: seeding infinity in diffusion transformer towards generic video restoration.

## For Beginners

SeedVR is a "Swiss army knife" for video restoration. Unlike models
that only handle upscaling or only denoising, SeedVR can fix many types of video
degradation using a single model, powered by a large transformer architecture that
learned from millions of videos what clean footage should look like.

**Usage:**

## How It Works

SeedVR (Wang et al., 2025) uses a Diffusion Transformer (DiT) for generic video restoration:

- DiT backbone: replaces the traditional U-Net with transformer blocks for better scaling
- Shifted window attention: efficient 3D (spatio-temporal) self-attention with linear

complexity, enabling processing of long video sequences

- Text-to-video priors: initialized from pretrained T2V model, providing strong knowledge

of natural video appearance and motion

- Generic restoration: handles SR, denoising, deblurring, and compression artifact removal

within a single unified model by learning to reverse various degradations

**Reference:** "SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic
Video Restoration" (Wang et al., 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeedVR(NeuralNetworkArchitecture<>,SeedVROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SeedVR model in native training mode. |
| `SeedVR(NeuralNetworkArchitecture<>,String,SeedVROptions)` | Creates a SeedVR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

