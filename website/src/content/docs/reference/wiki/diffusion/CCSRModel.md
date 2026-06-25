---
title: "CCSRModel<T>"
description: "CCSR: Content-Consistent Super-Resolution with diffusion-based controllable restoration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

CCSR: Content-Consistent Super-Resolution with diffusion-based controllable restoration.

## For Beginners

Diffusion SR models sometimes "imagine" details that weren't
in the original image (hallucination). CCSR prevents this by adding consistency checks
at every step — ensuring the upscaled image always matches the original low-res version
when downscaled back. This gives sharper results without inventing fake details.

## How It Works

CCSR introduces a non-uniform timestep sampling strategy and a content-consistency
regularization to prevent the diffusion SR model from hallucinating incorrect details.
Uses a compact VAE refinement module to maintain pixel-level fidelity.

Technical specifications:

- Architecture: SD2.1 U-Net with content-consistency regularization
- Text encoder: OpenCLIP ViT-H/14 (1024-dim)
- Non-uniform timestep sampling: Focuses on structure-preserving timesteps
- Content-consistency: Downscale consistency check at each step
- Compact VAE refinement for pixel-level fidelity
- Guidance: 1.0 (content-consistency replaces CFG)

Reference: Sun et al., "Improving the Stability of Diffusion Models for Content Consistent Super-Resolution", 2024

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

