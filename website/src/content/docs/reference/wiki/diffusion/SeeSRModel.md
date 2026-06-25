---
title: "SeeSRModel<T>"
description: "SeeSR: Semantics-aware super-resolution using diffusion-based image upscaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.SuperResolution`

SeeSR: Semantics-aware super-resolution using diffusion-based image upscaling.

## For Beginners

Standard super-resolution can add wrong details (like putting
grass texture on a building). SeeSR first understands what's in each part of the image
(semantics), then generates appropriate details for each region. A face gets skin
texture, a building gets brick patterns, etc.

## How It Works

SeeSR extracts semantic tags from the low-resolution input and uses them to condition
the diffusion process, ensuring that generated high-frequency details are semantically
consistent. For example, it won't add brick texture to a face region.

Technical specifications:

- Architecture: SD2.1 U-Net with semantic tag extraction module
- Text encoder: OpenCLIP ViT-H/14 (1024-dim) for tag conditioning
- Input: 8 channels (4 latent + 4 LR image latent)
- Semantic tags: DAPE (Degradation-Aware Prompt Extractor)
- Training: SD2.1 backbone + semantic consistency loss
- Scale factor: 4x upscaling

Reference: Wu et al., "SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution", CVPR 2024

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

